import glob
import os

import cv2
import numpy as np
import torch
import torch.nn
import torch.nn.functional
import torchvision.utils

import model
import utility
from data.common import np2Tensor
from option import args


class TestMAnet():
    def __init__(self, args):
        super(TestMAnet, self).__init__()

        torch.manual_seed(args.seed)
        self.ckp = utility.checkpoint(args)
        self.args = args
        if self.ckp.ok:
            # loader = data.Data(args)
            self.model = model.Model(args, self.ckp)
            # self.loss = loss.Loss(args, self.ckp) if not args.test_only else None
            # self.t = Trainer(args, loader, model, loss, self.ckp)
            psnr = 0.0

    def test_global(self):
        hr_dir = "Your/path/to/SRTestset"
        # hr_dir = "datasets"
        scale = self.args.scale[0]
        test_dataset_names = ["Set5", "Set14", "B100", "Urban100"]
        # test_dataset_names = ["Set14"]
        self.model.eval()
        for test_datset_name in test_dataset_names:
            test_hr_si_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale), "HR_si")
            test_lr_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale))
            test_si_hr = sorted(glob.glob(os.path.join(test_hr_si_dir, "*.png")))
            with torch.no_grad():
                test_mode = 0
                for noi in [self.args.noise]:
                    eval_psnr = 0
                    eval_ssim = 0
                    test_lr_dir_aim = os.path.join(test_lr_dir, "LR_mode%d_noise%d" % (test_mode, noi))
                    self.ckp.write_log(test_lr_dir_aim)
                    _lr = sorted(glob.glob(os.path.join(test_lr_dir_aim, "*.png")))
                    for id, filename in enumerate(test_si_hr):
                        img_hr = cv2.imread(filename)[:, :, ::-1]  # RGB
                        img_lr = cv2.imread(_lr[id])[:, :, ::-1]  # RGB

                        img_hr_tensor = np2Tensor(img_hr).unsqueeze(0)
                        img_hr_tensor = self.crop_border2(img_hr_tensor, scale).cuda()
                        img_lr_tensor = np2Tensor(img_lr).unsqueeze(0).cuda()

                        if self.args.model == 'blindsrMANet':
                            sr, _ = self.model(img_lr_tensor)
                        else:
                            sr = self.model(img_lr_tensor)

                        sr = utility.quantize(sr, self.args.rgb_range)
                        hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                        # metrics
                        eval_psnr += utility.calc_psnr(
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
                            0,
                            test_datset_name,
                            scale,
                            noi,
                            eval_psnr / len(_lr),
                            eval_ssim / len(_lr),
                        ))

    def test_global_lpips(self):
        import lpips
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        hr_dir = "SRTestset"
        scale = self.args.scale[0]
        test_dataset_names = ["Set5", "Set14", "B100", "Urban100"]
        # test_dataset_names = ["Set14"]
        self.model.eval()
        for test_datset_name in test_dataset_names:
            test_hr_si_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale), "HR_si")
            test_lr_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale))
            test_si_hr = sorted(glob.glob(os.path.join(test_hr_si_dir, "*.png")))
            with torch.no_grad():
                test_mode = 0
                for noi in [self.args.noise]:
                    lpips_v = 0

                    test_lr_dir_aim = os.path.join(test_lr_dir, "LR_mode%d_noise%d" % (test_mode, noi))
                    self.ckp.write_log(test_lr_dir_aim)
                    _lr = sorted(glob.glob(os.path.join(test_lr_dir_aim, "*.png")))

                    for id, filename in enumerate(test_si_hr):
                        img_hr = cv2.imread(filename)[:, :, ::-1]  # RGB
                        img_lr = cv2.imread(_lr[id])[:, :, ::-1]  # RGB
                        img_hr_tensor = np2Tensor(img_hr).unsqueeze(0).cuda()
                        # img_hr_tensor = self.crop_border2(img_hr_tensor, scale).cuda()
                        img_lr_tensor = np2Tensor(img_lr).unsqueeze(0).cuda()

                        if self.args.model == 'blindsrMANet':
                            sr, _ = self.model(img_lr_tensor)
                        else:
                            sr = self.model(img_lr_tensor)

                        # sr = utility.quantize(sr, self.args.rgb_range)
                        # hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                        sr = sr.clamp(0, 255) / 255.0 * 2.0 - 1.0
                        img_hr_tensor = img_hr_tensor.clamp(0, 255) / 255.0 * 2.0 - 1.0

                        lpips_v += loss_fn_alex(img_hr_tensor, sr).squeeze().cpu().numpy()

                        # sr = utility.quantize(sr, self.args.rgb_range)
                        # hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                        # metrics
                        # eval_psnr += utility.calc_psnr(
                        #     sr, hr, scale, self.args.rgb_range,
                        #     benchmark=True
                        # )
                        # eval_ssim += utility.calc_ssim(
                        #     sr, hr, scale,
                        #     benchmark=True
                        # )
                        # ckp.log[-1, idx_scale] = eval_psnr / len(loader_test)
                    self.ckp.write_log(
                        '[Epoch {}---{} x{} SI Noise {:.1f}]\t LPIPS: {:.3f}'.format(
                            0,
                            test_datset_name,
                            scale,
                            noi,
                            lpips_v / len(_lr),
                        ))

    def test_unseen_global(self):
        hr_dir = "SRTestset"
        # hr_dir = "/test/DASR/datasets"
        scale = self.args.scale[0]
        test_dataset_names = ["Urban100", "Set5", "Set14", "B100"]
        # test_dataset_names = ["Set14"]
        self.model.eval()
        for test_datset_name in test_dataset_names:
            test_hr_si_dir = os.path.join(hr_dir, "%s_ub0515sn_x%d" % (test_datset_name, scale), "HR_si")
            test_lr_dir = os.path.join(hr_dir, "%s_ub0515sn_x%d" % (test_datset_name, scale))
            test_si_hr = sorted(glob.glob(os.path.join(test_hr_si_dir, "*.png")))
            with torch.no_grad():
                test_mode = 0
                for noi in [self.args.noise]:
                    eval_psnr = 0
                    eval_ssim = 0
                    test_lr_dir_aim = os.path.join(test_lr_dir, "LR_mode%d_noise%d" % (test_mode, noi))
                    self.ckp.write_log(test_lr_dir_aim)
                    _lr = sorted(glob.glob(os.path.join(test_lr_dir_aim, "*.png")))
                    for id, filename in enumerate(test_si_hr):
                        img_hr = cv2.imread(filename)[:, :, ::-1]  # RGB
                        img_lr = cv2.imread(_lr[id])[:, :, ::-1]  # RGB

                        img_hr_tensor = np2Tensor(img_hr).unsqueeze(0)
                        img_hr_tensor = self.crop_border2(img_hr_tensor, scale).cuda()
                        img_lr_tensor = np2Tensor(img_lr).unsqueeze(0).cuda()

                        if self.args.model == 'blindsrMANet':
                            sr, _ = self.model(img_lr_tensor)
                        else:
                            sr = self.model(img_lr_tensor)

                        sr = utility.quantize(sr, self.args.rgb_range)
                        hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                        # metrics
                        eval_psnr += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=True
                        )
                        eval_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=True
                        )
                        # ckp.log[-1, idx_scale] = eval_psnr / len(loader_test)
                    self.ckp.write_log(
                        '[Epoch {}---{} x{} SI Noise {:.1f}]\tPSNR: {:.2f} SSIM: {:.4f}'.format(
                            0,
                            test_datset_name,
                            scale,
                            noi,
                            eval_psnr / len(_lr),
                            eval_ssim / len(_lr),
                        ))

    def crop_border2(self, img_hr, scale):
        b, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :int(h // scale * scale), :int(w // scale * scale)]

        return img_hr

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.args.scale)))
        self.model.eval()

        timer_test = utility.timer()
        hr_dir = "SRTestset"
        scale = self.args.scale[0]

        b100_hr_sv_dir = os.path.join(hr_dir, "B100_x%d" % scale, "HR_sv")
        b100_lr_dir = os.path.join(hr_dir, "B100_x%d" % scale)

        b100_sv_hr = sorted(glob.glob(os.path.join(b100_hr_sv_dir, "*.png")))

        self.ckp.write_log('Evaluation on spatial variant:')
        test_times = 0
        test_para = [1, 2, 3, 4, 5]
        psrn_ave = 0.0
        with torch.no_grad():
            for noi in [self.args.noise]:
                for ts in test_para:
                    test_times += 1
                    eval_psnr = 0
                    eval_ssim = 0
                    test_lr_dir = os.path.join(b100_lr_dir, "LR_mode%d_noise%d" % (ts, noi))
                    # test_lr_dir = os.path.join(b100_lr_dir, "mode_%d_loc_noise_%d_lr_n10_b40" % (ts, noi))
                    self.ckp.write_log(test_lr_dir)
                    b100_lr = sorted(glob.glob(os.path.join(test_lr_dir, "*.png")))

                    for id, filename in enumerate(b100_sv_hr):
                        img_hr = cv2.imread(filename)[:, :, ::-1]
                        img_lr = cv2.imread(b100_lr[id])[:, :, ::-1]
                        img_hr_tensor = np2Tensor(img_hr).unsqueeze(0)
                        img_hr_tensor = self.crop_border2(img_hr_tensor, scale).cuda()
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
                        eval_psnr += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=True
                        )
                        eval_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=True
                        )

                        # ckp.log[-1, idx_scale] = eval_psnr / len(loader_test)
                    self.ckp.write_log(
                        '[Epoch {}---{} x{} SV mode {:.1f}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                            0,
                            "B100",
                            scale,
                            ts,
                            eval_psnr / len(b100_lr),
                            eval_ssim / len(b100_lr),
                        ))



    def gen_u100_imgs(self):
        hr_dir = "SRTestset"
        save_dir = "results/"

        # test_mth = "Ours_noP_x2"
        # test_mth = "Ours_x2"
        # test_mth = "Ours_noL_x2"
        # test_mth = "Ours_nosa_x2"
        # test_mth = "Ours_nocb_x2"
        test_mth = "Urban100Bic"

        # hr_dir = "datasets"
        scale = self.args.scale[0]
        test_dataset_names = ["Urban100"]

        targe_id = 92
        targe_id = (targe_id - 1) * 9 + 0

        # test_dataset_names = ["Set14"]
        self.model.eval()
        for test_datset_name in test_dataset_names:
            test_hr_si_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale), "HR_si")
            test_lr_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale))
            test_si_hr = sorted(glob.glob(os.path.join(test_hr_si_dir, "*.png")))

            save_dir = os.path.join(save_dir, test_mth, test_datset_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with torch.no_grad():
                test_mode = 0
                for noi in [self.args.noise]:
                    test_lr_dir_aim = os.path.join(test_lr_dir, "LR_mode%d_noise%d" % (test_mode, noi))
                    self.ckp.write_log(test_lr_dir_aim)
                    _lr = sorted(glob.glob(os.path.join(test_lr_dir_aim, "*.png")))
                    for id, filename in enumerate(test_si_hr):
                        if id != targe_id:
                            continue
                        img_hr = cv2.imread(filename)[:, :, ::-1]  # RGB
                        img_lr = cv2.imread(_lr[id])[:, :, ::-1]  # RGB

                        img_hr_tensor = np2Tensor(img_hr).unsqueeze(0)
                        img_hr_tensor = self.crop_border2(img_hr_tensor, scale).cuda()
                        img_lr_tensor = np2Tensor(img_lr).unsqueeze(0).cuda()

                        if self.args.model == 'blindsrMANet':
                            srtns, _ = self.model(img_lr_tensor)
                        else:
                            srtns = self.model(img_lr_tensor)

                        sr = utility.quantize(srtns, self.args.rgb_range)
                        hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                        # metrics
                        psnr = utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=True
                        )
                        img_name = filename.split('/')[-1]
                        img_name = img_name.split('.png')[0]

                        torchvision.utils.save_image(srtns / 255.0, os.path.join(save_dir, "%s_%s_%.2f.png" % (
                        img_name, test_mth, psnr)))

    def gen_b100_imgs(self):
        hr_dir = "SRTestset"
        save_dir = "results/"

        # test_mth = "Ours_noP_x2"
        # test_mth = "Ours_x2"
        # test_mth = "Ours_noL_x2"
        # test_mth = "Ours_nosa_x2"
        # test_mth = "Ours_nocb_x2"
        # test_mth = "Ours"
        test_mth = "B100Bic"

        # hr_dir = "datasets"
        scale = self.args.scale[0]
        test_dataset_names = ["B100"]

        # targe_id = 56
        # targe_id = (targe_id-1)*9 + 7

        # test_dataset_names = ["Set14"]
        self.model.eval()
        for test_datset_name in test_dataset_names:
            test_hr_si_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale), "HR_si")
            test_lr_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale))
            test_si_hr = sorted(glob.glob(os.path.join(test_hr_si_dir, "*.png")))

            save_dir = os.path.join(save_dir, test_mth, test_datset_name + "x%d" % scale)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with torch.no_grad():
                test_mode = 0
                for noi in [self.args.noise]:
                    test_lr_dir_aim = os.path.join(test_lr_dir, "LR_mode%d_noise%d" % (test_mode, noi))
                    self.ckp.write_log(test_lr_dir_aim)
                    _lr = sorted(glob.glob(os.path.join(test_lr_dir_aim, "*.png")))
                    for id, filename in enumerate(test_si_hr):
                        # if id != targe_id:
                        #     continue
                        img_hr = cv2.imread(filename)[:, :, ::-1]  # RGB
                        img_lr = cv2.imread(_lr[id])[:, :, ::-1]  # RGB

                        img_hr_tensor = np2Tensor(img_hr).unsqueeze(0)
                        img_hr_tensor = self.crop_border2(img_hr_tensor, scale).cuda()
                        img_lr_tensor = np2Tensor(img_lr).unsqueeze(0).cuda()

                        if self.args.model == 'blindsrMANet':
                            srtns, _ = self.model(img_lr_tensor)
                        else:
                            srtns = self.model(img_lr_tensor)

                        sr = utility.quantize(srtns, self.args.rgb_range)
                        hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                        # metrics
                        psnr = utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=True
                        )
                        img_name = filename.split('/')[-1]
                        img_name = img_name.split('.png')[0]

                        torchvision.utils.save_image(srtns / 255.0, os.path.join(save_dir, "%s_%s_%.2f.png" % (
                        img_name, test_mth, psnr)))

    def gen_real_imgs(self):
        hr_dir = "SRTestset/historical/"
        save_dir = "results/"

        # test_mth = "Ours_noP_x2"
        # test_mth = "Ours_x2"
        # test_mth = "Ours_noL_x2"
        # test_mth = "Ours_nosa_x2"
        # test_mth = "Ours_nocb_x2"
        # test_mth = "DASR"
        test_mth = "DASR"

        # hr_dir = "datasets"
        scale = self.args.scale[0]
        test_dataset_names = ["historical"]

        # targe_id = 56
        # targe_id = (targe_id-1)*9 + 7

        # test_dataset_names = ["Set14"]
        self.model.eval()
        for test_datset_name in test_dataset_names:
            # test_hr_si_dir = os.path.join(hr_dir, "%s_x%d" % (test_datset_name, scale), "HR_si")
            test_lr_dir = os.path.join(hr_dir, "LR")
            test_si_lr = sorted(glob.glob(os.path.join(test_lr_dir, "*.png")))

            save_dir = os.path.join(save_dir, test_mth, test_datset_name + "x%d" % scale)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with torch.no_grad():
                test_mode = 0
                for noi in [self.args.noise]:

                    test_lr_dir_aim = test_lr_dir
                    self.ckp.write_log(test_lr_dir_aim)
                    _lr = sorted(glob.glob(os.path.join(test_lr_dir_aim, "*.png")))
                    for id, filename in enumerate(test_si_lr):
                        # if id != targe_id:
                        #     continue
                        # img_hr = cv2.imread(filename)[:, :, ::-1]  # RGB
                        img_lr = cv2.imread(_lr[id])[:, :, ::-1]  # RGB

                        # img_hr_tensor = np2Tensor(img_hr).unsqueeze(0)
                        # img_hr_tensor = self.crop_border2(img_hr_tensor, scale).cuda()
                        img_lr_tensor = np2Tensor(img_lr).unsqueeze(0).cuda()

                        if self.args.model == 'blindsrMANet':
                            srtns, _ = self.model(img_lr_tensor)
                        else:
                            srtns = self.model(img_lr_tensor)

                        # sr = utility.quantize(srtns, self.args.rgb_range)
                        # hr = utility.quantize(img_hr_tensor, self.args.rgb_range)

                        # metrics
                        # psnr = utility.calc_psnr(
                        #     sr, hr, scale, self.args.rgb_range,
                        #     benchmark=True
                        # )
                        img_name = filename.split('/')[-1]
                        img_name = img_name.split('.png')[0]

                        torchvision.utils.save_image(srtns / 255.0,
                                                     os.path.join(save_dir, "%s_%s.png" % (img_name, test_mth)))

    def test_runtime(self):
        scale = self.args.scale[0]
        self.model.eval()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        test_results = []
        sum_test_rumtime = 0.0
        with torch.no_grad():
            for _ in range(11):
                img_lr_tensor = torch.rand(1, 3, 48, 48).cuda()
                start.record()
                srtns = self.model(img_lr_tensor)
                end.record()
                torch.cuda.synchronize()
                t_ = start.elapsed_time(end)
                test_results.append(t_)  # milliseconds
                # sum_test_rumtime += t_
            print(test_results)
            # print(sum_test_rumtime / (len(test_results)-1))
            print(np.mean(test_results[1:]))


if __name__ == '__main__':
    tm = TestMAnet(args)
    # tm.test_global()
    # tm.gen_u100_imgs()
    tm.test_unseen_global()
    # tm.test_global_lpips()
    # tm.gen_u100_imgs()
    # tm.test_runtime()
    # tm.test()
    # tm.testAproxLocal()
