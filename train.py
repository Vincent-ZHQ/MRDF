import argparse

import toml
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset.fakeavceleb import FakeavcelebDataModule
# from dataset.dfdc import DFDCDataModule
from model import AVDF, AVDF_Ensemble, AVDF_Multilabel, AVDF_Multiclass, MRDF_Margin, MRDF_CE
from src.utils import LrLogger, EarlyStoppingLR

import os, time, random
import numpy as np
import logging

# log recorder
def set_log(args):

    logs_dir = os.path.join(args.outputs, 'log')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file_path = os.path.join(logs_dir, f'{args.save_name}-{log_name_time}.log')

    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


parser = argparse.ArgumentParser(description="MRDF training")
parser.add_argument("--dataset", type=str, default='fakeavceleb')
parser.add_argument("--model_type", type=str, default='MRDF_CE')
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--precision", default=16)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=None)
parser.add_argument("--max_epochs", type=int, default=30)
parser.add_argument("--min_epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=0)
parser.add_argument("--log_steps", type=int, default=20)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save_name", type=str, default='model')
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--margin_audio", type=float, default=0.0)
parser.add_argument("--margin_visual", type=float, default=0.0)
parser.add_argument("--margin_contrast", type=float, default=0.0)
parser.add_argument("--outputs", type=str, default='/Path/TO/outputs/')

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key])
    return dst_str

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    args = parser.parse_args()
    set_seed(42)
    
    logger = set_log(args)

    learning_rate = args.learning_rate 
    weight_decay = args.weight_decay
    dataset = args.dataset

    print("pytorch version: ", torch.__version__)
    print("cuda version: ", torch.version.cuda)
    print("cudnn version: ", torch.backends.cudnn.version())
    print("gpu name: ", torch.cuda.get_device_name())
    print("gpu index: ", torch.cuda.current_device())

    results = []

    model_dict = {'AVDF': AVDF, 'AVDF_Ensemble': AVDF_Ensemble, 'AVDF_Multilabel': AVDF_Multilabel, 'AVDF_Multiclass': AVDF_Multiclass, 'MRDF_Margin': MRDF_Margin, 'MRDF_CE': MRDF_CE} 
    for train_fold in ['train_1.txt', 'train_2.txt', 'train_3.txt', 'train_4.txt', 'train_5.txt']:
    # for train_fold in ['train_5.txt', 'train_1.txt']:
        args.save_name_id = args.save_name + '_' + train_fold[:-4]

        model = model_dict[args.model_type](
            margin_contrast=args.margin_contrast,
            margin_audio=args.margin_audio,
            margin_visual=args.margin_visual,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            distributed=args.gpus > 1
        )


        dm = FakeavcelebDataModule(
            root=args.data_root,
            train_fold = train_fold,
            batch_size=args.batch_size, num_workers=args.num_workers,
            take_train=args.num_train, take_dev=args.num_val,
        )

        try:
            precision = int(args.precision)
        except ValueError:
            precision = args.precision

        monitor = "val_re"
        early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=args.patience, verbose=False, mode="max")

        trainer = Trainer(log_every_n_steps=args.log_steps, precision=precision, min_epochs=args.min_epochs, max_epochs=args.max_epochs,
            callbacks=[
                ModelCheckpoint(
                    dirpath=f"{args.outputs}/ckpts/{args.model_type}", save_last=False,
                    filename=args.model_type + '_' + args.save_name_id + '_' + "{epoch}-{val_loss:.3f}",
                    monitor=monitor, mode="max"
                ),
                LrLogger(),
                EarlyStoppingLR(lr_threshold=1e-7),
                early_stop_callback

            ], enable_checkpointing=True,
            benchmark=True,
            num_sanity_val_steps=0,
            deterministic='warn',
            accelerator="auto",
            devices=args.gpus,
            strategy=None if args.gpus < 2 else "ddp",
            resume_from_checkpoint=args.resume,
        )

        # print(args.learning_rate, args.weight_decay, args.margin_audio, args.margin_visual, args.margin_contrast)
        trainer.fit(model, dm)

        # test
        model.eval()
        # result = trainer.test(model, dm.val_dataloader()) # , ckpt_path="best"
        result = trainer.test(model, dm.test_dataloader(), ckpt_path="best")
        results.append(result)
        print(result)
        logger.info('Result of ' + train_fold + ': ' + dict_to_str(result[0]))

    def dict_mean(dict_list):
        mean_dict = {}
        for key in dict_list[0][0].keys():
            mean_dict[key] = sum(d[0][key] for d in dict_list) / len(dict_list)
        return mean_dict


    print(results)
    print(dict_mean(results))
    logger.info('Final Average Performance: ' + dict_to_str(dict_mean(results)))



