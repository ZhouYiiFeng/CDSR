from importlib import import_module

from dataloader import MSDataLoader


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())  ## load the right dataset loader module
            trainset = getattr(module_train, args.data_train)(
                args)  ## load the dataset, args.data_train is the  dataset name
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test, train=False)
        elif args.data_test in ['g_x2', 'g_x2', 'g_x4', 'iso_x2']:
            module_test = import_module('data.patch_benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,
                                                        test_datasets_name=args.test_datasets_name,
                                                        gmode=args.gmode,
                                                        gnoise=args.gnoise,
                                                        train=False)
        else:
            module_test = import_module('data.' + args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )
