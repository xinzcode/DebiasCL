import numpy as np
import torch


class CL(torch.nn.Module):
    def __init__(self, data):
        super(CL, self).__init__()
        self.dev = data.dev
        self.r = data.R

    def calculate_loss(self, x, y, masks_x, masks_y):

        cl_loss = self.sim_loss(x, masks_x, y, masks_y, self.r)

        return cl_loss

    def sim_loss(self, x, word_masks, y, img_masks, temperature):
        batch_size = x.size(0)

        out_1 = AvgPoolSequence(word_masks, x)
        out_2 = AvgPoolSequence(img_masks, y)

        # 分子： *为对应位置相乘，也是点积
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
        # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
        # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
        # 加上exp操作，该操作实际计算了分母
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).byte()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # print(pos_sim)
        # print(sim_matrix.sum(dim=-1))
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


class DCL(torch.nn.Module):
    def __init__(self, data):
        super(DCL, self).__init__()
        self.device = data.dev
        self.r = data.R
        self.debiased = True

    def calculate_loss(self, x, y, masks_x, masks_y):

        cl_loss = self.sim_loss(x, masks_x, y, masks_y, self.r)

        return cl_loss

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size)).byte()
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def sim_loss(self, x, word_masks, y, img_masks, temperature):
        tau_plus = 0.1
        batch_size = x.size(0)

        out_1 = AvgPoolSequence(word_masks, x)
        out_2 = AvgPoolSequence(img_masks, y)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if self.debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss


def AvgPoolSequence(attn_mask, feats, e=1e-12):
    """ The function will average pool the input features 'feats' in
        the second to rightmost dimension, taking into account
        the provided mask 'attn_mask'.
    Inputs:
        attn_mask (torch.Tensor): [batch_size, ...x(N), 1] Mask indicating
                                  relevant (1) and padded (0) positions.
        feats (torch.Tensor): [batch_size, ...x(N), D] Input features.
    Outputs:
        feats (torch.Tensor) [batch_size, ...x(N-1), D] Output features
    """

    length = attn_mask.sum(-1)
    # pool by word to get embeddings for a sequence of words
    mask_words = attn_mask.float() * (1 / (length.float().unsqueeze(-1).expand_as(attn_mask) + e))
    feats = feats * mask_words.unsqueeze(-1).expand_as(feats)
    feats = feats.sum(dim=-2)

    return feats
