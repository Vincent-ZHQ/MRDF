Code for Cross-Modality and Within-Modality Regularization for Audio-Visual DeepFake Detection. (ICASSP 2024)

## Environemt
Python=3.8, Pytorch=1.13, pytorch_lightning==1.7.7, CUDA=11.6
```
conda create -n df116 python=3.8
pip install -r requirements.txt
```

## Code Structure
```
-- dataset (FakeAVCeleb)
-- model: 
    avhubert (download from https://github.com/facebookresearch/av_hubert/tree/main/avhubert) with some small changes, ImageEncoder.py, model.py
    model realted files (__init__, avdf, avdf_ensemble, avdf_multiclass, avdf_multilabel, mrdf_margin, mrdf_ce)
-- fairseq: (download from https://github.com/facebookresearch/fairseq with some small changes)
-- outputs: log, results, ckpts
-- data/FakeAVCeleb_v1.2: path to the dataset (FakeAVCeleb) and splits (5 folds)
-- utils: loss, figure, utils
-- main.py
-- train.py
-- test.py
-- requirements.txt
```

## Data Preparation
[FakeAVCeleb](https://sites.google.com/view/fakeavcelebdash-lab/) is an audio-visual deepfake detection dataset. FakeAVCeleb consists of 500 real videos and over 20,000 fake videos, spanning five ethnic groups, each with 100 real videos from 100 subjects. As there are no official split methods, we provide a balanced split method with a 1:1:1:1 ratio across four categories,  FakeAudio-FakeVideo (FAFV), FakeAudio-RealVideo (FARV), RealAudio-FakeVideo (RAFV), and RealAudio-RealVideo (RARV), and use 500 video from each category with one video from one subject for each category. In total, we have 2000 videos and split them into 5 folds and utilize a subject-independent 5-fold-cross-validation strategy for equitable comparisons. The subjects for testing are not seen during training for each fold.

## Train and Test

 For direct inference on Food101, we provide the pretrained checkpoint example [here](https://drive.google.com/drive/folders/1WdfkPRlzX-3Y4xSSS7F9WrR_Hdqpdudw?usp=sharing).

```
## train example
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type MRDF_CE --save_name MRDF_CE \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb

CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type MRDF_Margin --save_name MRDF_Margin \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb

## test example
CUDA_VISIBLE_DEVICES=0 python test.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/MRDF_Margin_train_2.ckpt \
  --train_fold train_2.txt --model_type MRDF_Margin

CUDA_VISIBLE_DEVICES=0 python test.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/MRDF_CE_train_2.ckpt \
  --train_fold train_2.txt --model_type MRDF_CE
```
More examples could be seen in sript/train.sh and sript/test.sh. Due to the environment difference, the results could be a little different with those we reported in the paper. 



## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```text

```

## Acknowledgements

Some code is borrowed from 
[ControlNet/LAV-DF](https://github.com/ControlNet/LAV-DF) and 
[facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert).
