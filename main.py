import random
import re
import time

import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

from bert_crf import MMEncoder, Decoder, 
from cl import CL, DCL
from data import get_data_loader, CustomDataset, Data, bert_encode, char_is_emoji
from warmup_scheduler import GradualWarmupScheduler

from utils.metric import get_ner_fmeasure


seed_num = 1
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)



if __name__ == '__main__':
    torch.set_printoptions(threshold=np.inf)
    # data
    data = Data()
    train_loader, dev_loader, test_loader = get_data_loader(data)

    if data.status == 'train':
        # model
        encoder = MMEncoder(data)
        encoder.to(data.dev)
        decoder = Decoder(data)
        decoder.to(data.dev)
        cl = DCL(data)
        cl.to(data.dev)

        # optimizer_encoder
        word_bert_params = list(map(id, encoder.word_bert.parameters()))
        word_gru_params = list(map(id, encoder.word_gru.parameters()))
        gat_params = list(map(id, encoder.gat.parameters()))
        fu_params = list(map(id, encoder.fusion.parameters()))
        rest_params = filter(lambda p: id(p) not in word_bert_params + word_gru_params + gat_params + fu_params,
                             encoder.parameters())
        params_encoder = [
            {"params": encoder.word_bert.parameters(), "lr": data.LEARNING_RATE_WORD_BERT},
            {"params": encoder.word_gru.parameters(), "lr": data.LEARNING_RATE_WORD_GRU},
            {"params": encoder.gat.parameters(), "lr": data.LEARNING_RATE_WORD_GAT},
            {"params": encoder.fusion.parameters(), "lr": data.LEARNING_RATE_FUSION},
            {"params": rest_params, "lr": data.LEARNING_RATE_BASE},
        ]
        optimizer_encoder = torch.optim.Adam(params_encoder, weight_decay=data.WEIGHT_DECAY)
        scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, 2, gamma=0.8)

        # optimizer_decoder
        params_decoder = [
            {"params": decoder.parameters(), "lr": data.LEARNING_RATE_BASE},
        ]
        optimizer_decoder = torch.optim.Adam(params_decoder, weight_decay=data.WEIGHT_DECAY)
        scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, 2, gamma=0.8)

        # train dev test
        best_p, best_r, best_f, best_f_p, best_f_r = -1, -1, -1, -1, -1
        dev_loss = -1
        for epoch in range(data.EPOCHS):
            start = time.time()
            encoder.train()
            decoder.train()

            for _, instance in enumerate(train_loader):
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
    
                ids = instance['ids']
                masks = instance['masks']
                targets = instance['tags']
                sent_lens = instance['sent_len']
                imgs = instance['imgs']
                obj_nums = instance['obj_nums']

                x, y, word_feat, img_feat, masks_x, masks_y = encoder(ids, masks, sent_lens, imgs)

                # normal loss
                cl_loss = cl.calculate_loss(x, y, masks_x, masks_y)

                sorted_obj_nums, sorted_indices = torch.sort(obj_nums, descending=False, dim=0)  # 递增排序
                ids_order = ids[sorted_indices]
                masks_order = masks[sorted_indices]
                targets_order = targets[sorted_indices]
                sent_lens_order = sent_lens[sorted_indices]
                imgs_order = imgs[sorted_indices]

                # hard loss
                hard_num = 5
                batch_size = ids.size(0)
                hard_ids = ids_order[batch_size-hard_num:, :]
                hard_masks = masks_order[batch_size-hard_num:, :]
                hard_targets = targets_order[batch_size-hard_num:, :]
                hard_sent_lens = sent_lens_order[batch_size-hard_num:]
                hard_imgs = imgs_order[batch_size-hard_num:, :]
                hard_x, hard_y, hard_word_feat, hard_img_feat, hard_masks_x, hard_masks_y = \
                    encoder(hard_ids, hard_masks, hard_sent_lens, hard_imgs)
                cl_loss_hard = cl.calculate_loss(hard_x, hard_y, hard_masks_x, hard_masks_y)

                # crf loss
                label_loss = decoder.calculate_loss(x, word_feat, masks_x, targets)

                loss = label_loss + cl_loss + 2 * cl_loss_hard

                if _ % 50 == 0:
                    print(f'Epoch: {epoch}, Batch: {_}, Loss:  {loss.item()}')

                loss.backward()
                optimizer_encoder.step()
                optimizer_decoder.step()

            scheduler_encoder.step()
            scheduler_decoder.step()


            acc, p, r, f = evaluate1(encoder, decoder, data, test_loader)
