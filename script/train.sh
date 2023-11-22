# train MRDF_CE
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type MRDF_CE --save_name MRDF_CE \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb

# train MRDF_Margin
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type MRDF_Margin --save_name MRDF_Margin \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb

# train AVDF
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type AVDF --save_name AVDF \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb

# train AVDF_ensemble
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type AVDF_Ensemble --save_name AVDF_Ensemble \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb

# train AVDF_multilabel
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type AVDF_Multilabel --save_name AVDF_Multilabel \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb

# train AVDF_multiclass
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type AVDF_Multiclass --save_name AVDF_Multiclass \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --dataset fakeavceleb