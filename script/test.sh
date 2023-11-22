# train MRDF_CE
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/MRDF_MarMRDF_CEgin_train_n.ckpt \
  --train_fold train_n.txt --model_type MRDF_CE

# test MRDF_Margin
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/MRDF_Margin_train_n.ckpt \
  --train_fold train_n.txt --model_type MRDF_Margin

# test AVDF
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/AVDF_train_n.ckpt \
  --train_fold train_n.txt --model_type AVDF

# test AVDF_ensemble
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/AVDF_ensemble_train_n.ckpt \
  --train_fold train_n.txt --model_type AVDF_ensemble

# test AVDF_multilabel
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/AVDF_multilabel_train_n.ckpt \
  --train_fold train_n.txt --model_type AVDF_multilabel

# test AVDF_multiclass
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_root ./data/FakeAVCeleb_v1.2/ \
  --checkpoint /Path/To/AVDF_multiclass_train_n.ckpt \
  --train_fold train_n.txt --model_type AVDF_multiclass