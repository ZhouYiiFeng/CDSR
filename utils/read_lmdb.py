# from trainer import Trainer
import os

import torch

import data
import loss
import model
import utility
from option import args
from trainer_ref import TrainerRF



if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = TrainerRF(args, loader, model, loss, checkpoint)
        psnr = 0.0

        for idx_img, (hr, filename, _) in enumerate(loader.loader_test):
            # hr = hr.cuda()  # b, n, c, h, w
            b, n, c, h, w = hr.shape
            # hr = crop_border(hr, scale)
            hr_cln = hr[:, :, c // 2:, :, :].contiguous()  # noisy front
            hr_noisy = hr[:, :, :c // 2, :, :].contiguous()  # cln latter

            hr = hr_cln
            hr = hr[:, 0, ...]  # b, c, h, w
