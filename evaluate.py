import argparse

import toml
import torch

from dataset.fakeavceleb import FakeavcelebDataModule
from inference import inference_batfd
from metrics import AP, AR
from model import MRDF
from post_process import post_process
from utils import read_json

parser = argparse.ArgumentParser(description="BATFD evaluation")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--train_fold", type=str, default='train_1.txt')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--modalities", type=str, nargs="+", default=["fusion"])
parser.add_argument("--subset", type=str, nargs="+", default=["full"])
parser.add_argument("--gpus", type=int, default=1)


def visual_subset_condition(meta):
    return not (meta.modify_video is False and meta.modify_audio is True)


def audio_subset_condition(meta):
    return not (meta.modify_video is True and meta.modify_audio is False)


conditions = {
    "full": None,
    "subset_for_visual_only": visual_subset_condition,
    "subset_for_audio_only": audio_subset_condition
}


def evaluate_lavdf(config, args):
    for modal in args.modalities:
        assert modal in ["fusion", "audio", "visual"]

    for subset in args.subset:
        assert subset in ["full", "subset_for_visual_only", "subset_for_audio_only"]

    model_name = config["name"]
    alpha = config["soft_nms"]["alpha"]
    t1 = config["soft_nms"]["t1"]
    t2 = config["soft_nms"]["t2"]

    model_type = config["model_type"]
    v_feature = None
    a_feature = None

    # prepare model
    if config["model_type"] == "batfd_plus":
        model = BatfdPlus.load_from_checkpoint(args.checkpoint)
        require_match_scores = True
        get_meta_attr = BatfdPlus.get_meta_attr
    elif config["model_type"] == "batfd":
        model = MRDF.load_from_checkpoint(args.checkpoint)
    else:
        raise ValueError("Invalid model type")

    # prepare dataset
    dm = FakeavcelebDataModule(
        root=args.data_root,
        feature_types=(v_feature, a_feature),
        train_fold=args.train_fold,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    dm.setup()

    # inference and save dense proposals as csv file
    results = inference_batfd(model_name, model, dm, config["max_duration"], model_type, args.modalities, args.gpus)
    print(results)

if __name__ == '__main__':
    args = parser.parse_args()
    config = toml.load(args.config)
    torch.backends.cudnn.benchmark = True
    evaluate_lavdf(config, args)
