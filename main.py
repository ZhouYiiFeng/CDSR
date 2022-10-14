from option import args
import torch
import utility
import data
import model
import loss
from trainer import Trainer
import os


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        psnr = 0.0

        while not t.terminate():
            t.train()
            if t.epoch > args.epochs_encoder:
                test_psnr, log_dct = t.test3()
                for k, v in log_dct.items():
                    checkpoint.tb_logger.add_scalar(f'psnr/{k}', v, t.iteration)

                psnr = test_psnr
                target = t.model.get_model()
                model_dict = target.state_dict()
                keys = list(model_dict.keys())
                for key in keys:
                    if 'E.encoder_k' in key or 'queue' in key:
                        del model_dict[key]
                torch.save(
                    model_dict,
                    os.path.join(t.ckp.dir, 'model', 'model_{}_psnr_{:.2f}.pt'.format(t.epoch, test_psnr))
                )
        checkpoint.done()
