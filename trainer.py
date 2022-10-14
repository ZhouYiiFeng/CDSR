import os
from decimal import Decimal
import cv2
import glob
import torch
from data.common import np2Tensor
import torch.nn.functional as F
import torchvision.utils

import utility
# from utils import util_ma as util
from utils import util_ma as util


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.epoch = args.start_epoch
        self.ckp = ckp
        self.freeze_encoder = False
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        # self.model = torch.nn.DataParallel(my_model, range(self.args.n_GPUs))
        self.model = my_model
        # self.model_E = torch.nn.DataParallel(self.model.get_model().E, range(self.args.n_GPUs))
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.iteration = 0
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        self.epoch = epoch

        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        degrade = util.SRMDPreprocessing(
            scale=int(self.scale[0]),
            random=True,
            sample_mode=self.args.sample_mode,
            sv_mode=self.args.sv_mode,
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            noise=self.args.noise,
            is_training=True
        )

        timer = utility.timer()
        losses_contrast, losses_sr = utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            timer.tic()
            self.iteration += 1
            hr = hr.cuda()  # b, n, c, h, w
            lr, lr_n, b_kernels = degrade(hr)  # bn, c, h, w
            if self.args.noise != 0:
                lr = lr_n
            self.optimizer.zero_grad()

            # forward
            ## train degradation encoder
            if epoch <= self.args.epochs_encoder:
                # output, target = self.model_E(im_q=lr[:, 0, ...], im_k=lr[:, 1, ...])
                output, target = self.model.model.module.get_construct_learning_output(lr[:, 0, ...], lr[:, 1, ...])
                loss_constrast = self.contrast_loss(output, target)
                loss = loss_constrast

                losses_contrast.update(loss_constrast.item())

            ## train the whole network
            else:
                sr, output, target = self.model(lr)
                loss_SR = self.loss(sr, hr[:, 0, ...])
                loss_constrast = self.contrast_loss(output, target)
                loss = self.args.c_weight * loss_constrast + loss_SR

                losses_sr.update(loss_SR.item())
                losses_contrast.update(loss_constrast.item())

            if epoch > self.args.freeze_epoch:
                self.freeze_encoder = True
                self.args.c_weight = 0
                # self.ckp.write_log("############### Freeze the encoder. #############")
                self.model.model.module.freeze_encoder()


            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_contrast.avg,
                            timer.release()
                        ))
                if self.args.use_tb_logger:
                    self.ckp.tb_logger.add_scalar('contrast loss', losses_contrast.avg, self.iteration)
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg, losses_contrast.avg,
                            timer.release(),
                        ))
                if self.args.use_tb_logger:
                    self.ckp.tb_logger.add_scalar('contrast loss', losses_contrast.avg, self.iteration)
                    self.ckp.tb_logger.add_scalar('l1 loss', losses_sr.avg, self.iteration)

        self.loss.end_log(len(self.loader_train))

    def test(self, test_path="/your/data/path/SRTestset"):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        psrn_ave = 0.0

        timer_test = utility.timer()
        hr_dir = test_path

        scale = self.args.scale[0]
        s14_hr_si_dir = os.path.join(hr_dir, "Set14_x%d" % scale, "HR_si")
        b100_hr_sv_dir = os.path.join(hr_dir, "B100_x%d" % scale, "HR_sv")
        b100_lr_dir = os.path.join(hr_dir, "B100_x%d" % scale)
        s14_lr_dir = os.path.join(hr_dir, "Set14_x%d" % scale)

        s14_si_hr = sorted(glob.glob(os.path.join(s14_hr_si_dir, "*.png")))
        b100_sv_hr = sorted(glob.glob(os.path.join(b100_hr_sv_dir, "*.png")))

        self.ckp.write_log('Evaluation on spatial invariant:')
        test_times = 0
        rtn_log_dic = {}
        with torch.no_grad():
            test_mode = 0
            for noi in [self.args.noise]:
                test_times += 1
                eval_psnr = 0
                eval_ssim = 0
                test_lr_dir = os.path.join(s14_lr_dir, "LR_mode%d_noise%d" % (test_mode, noi))
                self.ckp.write_log(test_lr_dir)
                b100_lr = sorted(glob.glob(os.path.join(test_lr_dir, "*.png")))
                for id, filename in enumerate(s14_si_hr):
                    img_hr = cv2.imread(filename)[:, :, ::-1]
                    img_lr = cv2.imread(b100_lr[id])[:, :, ::-1]
                    img_hr_tensor = np2Tensor(img_hr).unsqueeze(0)
                    img_hr_tensor = self.crop_border(img_hr_tensor, scale).cuda()
                    img_lr_tensor = np2Tensor(img_lr).unsqueeze(0).cuda()

                    timer_test.tic()
                    if self.args.model == 'blindsrMANet':
                        sr, _ = self.model(img_lr_tensor)
                    else:
                        sr = self.model(img_lr_tensor)
                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr2(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=True
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=True
                    )

                    # ckp.log[-1, idx_scale] = eval_psnr / len(loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{} SI Noise {:.1f}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.epoch,
                        self.args.data_test,
                        scale,
                        noi,
                        eval_psnr / len(b100_lr),
                        eval_ssim / len(b100_lr),
                    ))
                psrn_ave += eval_psnr / len(b100_lr)
            psrn_ave = psrn_ave * 1.0 / test_times
            self.ckp.write_log(
                '[Epoch {}---{} x{} SI ave] \tPSNR: {:.3f}'.format(
                    self.epoch,
                    self.args.data_test,
                    scale,
                    psrn_ave
                ))
            psnr_si = psrn_ave
            rtn_log_dic["Set5_mode%d_noise%d" % (test_mode, noi)] = psnr_si

        self.ckp.write_log('Evaluation on spatial variant:')
        test_times = 0
        test_para = [1, 3, 5]
        psrn_ave = 0.0

        return psnr_si, rtn_log_dic


    def crop_border(self, img_hr, scale):
        b, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :int(h // scale * scale), :int(w // scale * scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr
