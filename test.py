import argparse
import toml
import torch

from pytorch_lightning import LightningModule, Trainer
from dataset.fakeavceleb import FakeavcelebDataModule
from model import AVDF, AVDF_Ensemble, AVDF_Multilabel, AVDF_Multiclass, MRDF_Margin, MRDF_CE

parser = argparse.ArgumentParser(description="MRDF evaluation")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--model_type", type=str, default='MRDF_CE')
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--train_fold", type=str, default='train_1.txt')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--gpus", type=int, default=1)

def inference_mrdf(model: LightningModule, dm: FakeavcelebDataModule,
    model_type: str, gpus: int = 1
):

    model.eval()

    test_dataset = dm.test_dataset

    trainer = Trainer(logger=False,
        enable_checkpointing=False, devices=1 if gpus > 1 else None,
        accelerator="gpu" if gpus > 0 else "cpu",
    )

    return trainer.test(model, dm.test_dataloader())

def evaluate_mrdf(args):

    model_dict = {'AVDF': AVDF, 'AVDF_Ensemble': AVDF_Ensemble, 'AVDF_Multilabel': AVDF_Multilabel, 'AVDF_Multiclass': AVDF_Multiclass, 'MRDF_Margin': MRDF_Margin, 'MRDF_CE': MRDF_CE} 

    model = model_dict[args.model_type].load_from_checkpoint(args.checkpoint)   
         
    # prepare dataset
    dm = FakeavcelebDataModule(
        root=args.data_root,
        train_fold=args.train_fold,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    dm.setup()

    # inference 
    results = inference_mrdf(model, dm, args.model_type, args.gpus)
    print(results)

if __name__ == '__main__':
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    evaluate_mrdf(args)
