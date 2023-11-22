from typing import Dict, List, Optional, Union, Sequence

import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.loss import ContrastLoss, MarginLoss


import fairseq
from fairseq.modules import LayerNorm
from fairseq.data.dictionary import Dictionary
import model.avhubert.hubert as hubert
import model.avhubert.hubert_pretraining as hubert_pretraining
import torchmetrics


def Average(lst):
    return sum(lst) / len(lst)


def Opposite(a):
    a = a + 1
    a[a>1.5] = 0
    return a


class AVDF_Ensemble(LightningModule):

    def __init__(self,
       margin_contrast=0.0, margin_audio=0.0, margin_visual=0.0, weight_decay=0.0001, learning_rate=0.0002, distributed=False
    ):
        super().__init__()
        self.model = hubert.AVHubertModel(cfg=hubert.AVHubertConfig,
                                          task_cfg=hubert_pretraining.AVHubertPretrainingConfig,
                                          dictionaries=hubert_pretraining.AVHubertPretrainingTask)

        self.embed = 768
        self.dropout = 0.1

        self.feature_extractor_audio_hubert = self.model.feature_extractor_audio
        self.feature_extractor_video_hubert = self.model.feature_extractor_video

        self.project_audio = nn.Sequential(LayerNorm(self.embed), nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))

        self.project_video = nn.Sequential(LayerNorm(self.embed), nn.Linear(self.embed, self.embed),
                                           nn.Dropout(self.dropout))

        self.project_hubert = nn.Sequential(self.model.layer_norm, self.model.post_extract_proj,
                                           self.model.dropout_input)

        self.final_proj_audio = self.model.final_proj
        self.final_proj_video = self.model.final_proj
        self.final_proj_hubert = self.model.final_proj

        self.fusion_encoder_hubert = self.model.encoder

        self.video_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        self.audio_classifier = nn.Sequential(nn.Linear(self.embed, 2))

        self.a_cls = CrossEntropyLoss()
        self.v_cls = CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auroc = torchmetrics.classification.BinaryAUROC(thresholds=None)
        self.f1score = torchmetrics.classification.BinaryF1Score()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.precisions = torchmetrics.classification.BinaryPrecision()

        self.best_loss = 1e9
        self.best_acc, self.best_auroc = 0.0, 0.0
        self.best_real_f1score, self.best_real_recall,  self.best_real_precision = 0.0, 0.0, 0.0
        self.best_fake_f1score, self.best_fake_recall, self.best_fake_precision = 0.0, 0.0, 0.0

        self.softmax = nn.Softmax(dim=1)

    def forward(self, video: Tensor, audio: Tensor, mask: Tensor):
        # print(audio.shape, video.shape)
        a_features = self.feature_extractor_audio_hubert(audio).transpose(1, 2)
        v_features = self.feature_extractor_video_hubert(video).transpose(1, 2)
        av_features = torch.cat([a_features, v_features], dim=2)

        a_cross_embeds = a_features.mean(1)
        v_cross_embeds = v_features.mean(1)

        a_features = self.project_audio(a_features)
        v_features = self.project_video(v_features)

        a_embeds = a_features.mean(1)
        v_embeds = v_features.mean(1)

        a_embeds = self.audio_classifier(a_embeds)
        v_embeds = self.video_classifier(v_embeds)

        return v_embeds, a_embeds

    def loss_fn(self, v_logits, a_logits, v_label, a_label) -> Dict[str, Tensor]:
        a_loss = self.a_cls(a_logits, a_label)
        v_loss = self.v_cls(v_logits, v_label)

        mm_loss = a_loss + v_loss
        loss = mm_loss 

        return {"loss": loss, "mm_loss": mm_loss}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:
        v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(v_logits, a_logits, batch['v_label'], batch['a_label'])

        # emsemble
        preds_a = torch.argmax(self.softmax(a_logits), dim=1)
        preds_v = torch.argmax(self.softmax(v_logits), dim=1)
        preds = preds_a + preds_v
        preds = torch.where(preds<2, 0, 1)

        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return {"loss": loss_dict["loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:

        v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(v_logits, a_logits, batch['v_label'], batch['a_label'])

        # emsemble
        preds_a = torch.argmax(self.softmax(a_logits), dim=1)
        preds_v = torch.argmax(self.softmax(v_logits), dim=1)
        preds = preds_a + preds_v
        preds = torch.where(preds<2, 0, 1)

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def test_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
                        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
                        ) -> Tensor:
        v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(v_logits, a_logits, batch['v_label'], batch['a_label'])

        # emsemble
        preds_a = torch.argmax(self.softmax(a_logits), dim=1)
        preds_v = torch.argmax(self.softmax(v_logits), dim=1)
        preds = preds_a + preds_v
        preds = torch.where(preds<2, 0, 1)

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}


    def training_step_end(self, training_step_outputs):
        train_acc = self.acc(training_step_outputs['preds'], training_step_outputs['targets']).item()
        train_auroc = self.auroc(training_step_outputs['preds'], training_step_outputs['targets']).item()
        
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_auroc", train_auroc, prog_bar=True)

    def validation_step_end(self, validation_step_outputs):
        val_acc = self.acc(validation_step_outputs['preds'], validation_step_outputs['targets']).item()
        val_auroc = self.auroc(validation_step_outputs['preds'], validation_step_outputs['targets']).item()
        
        self.log("val_re", val_acc+val_auroc, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_auroc", val_auroc, prog_bar=True)

    def training_epoch_end(self, training_step_outputs):
        train_loss = Average([i["loss"] for i in training_step_outputs]).item()
        preds = [item for list in training_step_outputs for item in list["preds"]]
        targets = [item for list in training_step_outputs for item in list["targets"]]
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        train_acc = self.acc(preds, targets).item()
        train_auroc = self.auroc(preds, targets).item()

        print("Train - loss:", train_loss, "acc: ", train_acc, "auroc: ", train_auroc)

    def validation_epoch_end(self, validation_step_outputs):
        valid_loss = Average([i["loss"] for i in validation_step_outputs]).item()
        preds = [item for list in validation_step_outputs for item in list["preds"]]
        targets = [item for list in validation_step_outputs for item in list["targets"]]
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        self.best_acc = self.acc(preds, targets).item()
        self.best_auroc = self.auroc(preds, targets).item()
        self.best_real_f1score = self.f1score(preds, targets).item()
        self.best_real_recall = self.recall(preds, targets).item()
        self.best_real_precision = self.precisions(preds, targets).item()

        self.best_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        self.best_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        self.best_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        self.best_loss = valid_loss
        print("Valid loss: ", self.best_loss, "acc: ", self.best_acc, "auroc: ", self.best_auroc,
              "real_f1score:",
              self.best_real_f1score, "real_recall: ", self.best_real_recall, "real_precision: ",
              self.best_real_precision, "fake_f1score: ", self.best_fake_f1score, "fake_recall: ",
              self.best_fake_recall, "fake_precision: ", self.best_fake_precision)

    def test_epoch_end(self, test_step_outputs):
        test_loss = Average([i["loss"] for i in test_step_outputs]).item()
        preds = [item for list in test_step_outputs for item in list["preds"]]
        targets = [item for list in test_step_outputs for item in list["targets"]]
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        test_acc = self.acc(preds, targets).item()
        test_auroc = self.auroc(preds, targets).item()
        test_real_f1score = self.f1score(preds, targets).item()
        test_real_recall = self.recall(preds, targets).item()
        test_real_precision = self.precisions(preds, targets).item()

        test_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        test_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        test_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        self.log("test_acc", test_acc)
        self.log("test_auroc", test_auroc)
        self.log("test_real_f1score", test_real_f1score)
        self.log("test_real_recall", test_real_recall)
        self.log("test_real_precision", test_real_precision)
        self.log("test_fake_f1score", test_fake_f1score)
        self.log("test_fake_recall", test_fake_recall)
        self.log("test_fake_precision", test_fake_precision)
        return {"loss": test_loss, "test_acc": test_acc, "auroc": test_auroc, "real_f1score": test_real_f1score,
                "real_recall": test_real_recall, "real_precision":
                    test_real_precision, "fake_f1score": test_fake_f1score, "fake_recall": test_fake_recall,
                "fake_precision": test_fake_precision}


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }
