from typing import Dict, List, Optional, Union, Sequence

import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import ContrastLoss, MarginLoss


import fairseq
from fairseq.modules import LayerNorm
from fairseq.data.dictionary import Dictionary
import model.avhubert.hubert as hubert
import model.avhubert.hubert_pretraining as hubert_pretraining
import torchmetrics

from sklearn.manifold import TSNE
from figure import visualize_tsne_2


def Average(lst):
    return sum(lst) / len(lst)


def Opposite(a):
    a = a + 1
    a[a>1.5] = 0
    return a


class MRDF(LightningModule):

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

        self.fusion_encoder_hubert = self.model.encoder

        self.final_proj_audio = self.model.final_proj
        self.final_proj_video = self.model.final_proj
        self.final_proj_hubert = self.model.final_proj

        # self.video_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        # self.audio_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        # #
        self.mm_classifier = nn.Sequential(nn.Linear(self.embed, self.embed), nn.ReLU(inplace=True),
                                              nn.Linear(self.embed, 2))

        self.contrast_loss = ContrastLoss(loss_fn=nn.CosineSimilarity(dim=-1), margin=margin_contrast)
        self.margin_loss_audio = MarginLoss(loss_fn=nn.CosineSimilarity(dim=-1), margin=margin_audio)
        self.margin_loss_visual = MarginLoss(loss_fn=nn.CosineSimilarity(dim=-1), margin=margin_visual)
        self.mm_cls = CrossEntropyLoss()
        # self.mm_cls = BCEWithLogitsLoss()
        # self.a_cls = CrossEntropyLoss()
        # self.v_cls = CrossEntropyLoss()

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
        av_features = self.project_hubert(av_features)

        a_embeds = a_features.mean(1)
        v_embeds = v_features.mean(1)

        # a_embeds = self.audio_classifier(a_embeds)
        # v_embeds = self.video_classifier(v_embeds)

        av_features, _ = self.fusion_encoder_hubert(av_features, padding_mask=mask)
        m_logits = self.mm_classifier(av_features[:, 0, :])

        # m_logits = a_embeds + v_embeds
        # m_logits = a_embeds

        return m_logits, v_cross_embeds, a_cross_embeds, v_embeds, a_embeds

    def get_avg_feat(self, feat, mask):
        mask_un = mask.to(dtype=torch.float).unsqueeze(1)
        feat = feat * mask_un
        mask_un_sum = torch.sum(mask_un, dim=1, dtype=torch.float)
        mask_un_sum[mask_un_sum == 0.] = 1.
        feat = torch.sum(feat, dim=1) / mask_un_sum
        return feat

    def loss_fn(self, m_logits, v_logits, a_logits, v_feats, a_feats, v_label, a_label, c_label, m_label) -> Dict[str, Tensor]:

        contrast_loss = self.contrast_loss(v_logits, a_logits, c_label)
        v_loss = self.margin_loss_visual(v_feats, v_label)
        a_loss = self.margin_loss_audio(a_feats, a_label)
        # a_loss = self.a_cls(a_feats, a_label)
        # v_loss = self.v_cls(v_feats, v_label)

        # print(m_logits.shape, m_label.shape)

        # add for multi-label
        # av_label = torch.stack((a_label, v_label), dim=1)
        # mm_loss = self.mm_cls(m_logits,  av_label.float())

        mm_loss = self.mm_cls(m_logits, m_label)

        # add for ensemble
        # mm_loss = a_loss + v_loss

        loss = mm_loss + a_loss + v_loss + contrast_loss #  # + contrast_loss # mm_loss + contrast_loss + a_loss + v_loss # + contrast_loss #

        # loss = mm_loss + contrast_loss
        return {"loss": loss, "mm_loss": mm_loss}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:
        m_logits, v_logits, a_logits, v_feats, a_feats = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_logits, a_logits, v_feats, a_feats, batch['v_label'], batch['a_label'],
                                               batch['c_label'], batch['m_label'])
        
        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        # emsemble
        # preds_a = torch.argmax(self.softmax(a_feats), dim=1)
        # preds_v = torch.argmax(self.softmax(v_feats), dim=1)
        # preds = preds_a + preds_v
        # preds = torch.where(preds<2, 0, 1)
        
        # multi-label
        # preds = torch.sigmoid(m_logits)
        # preds = torch.where(preds < 0.5, 0, 1)
        # preds = preds.sum(1)
        # preds = torch.where(preds<2, 0, 1)
        #
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return {"loss": loss_dict["loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:

        m_logits, v_logits, a_logits, v_feats, a_feats = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_logits, a_logits, v_feats, a_feats, batch['v_label'], batch['a_label'],
                                               batch['c_label'], batch['m_label'])
        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        # emsemble
        # preds_a = torch.argmax(self.softmax(a_feats), dim=1)
        # preds_v = torch.argmax(self.softmax(v_feats), dim=1)
        # preds = preds_a + preds_v
        # preds = torch.where(preds<2, 0, 1)

        # multi-label
        # preds = torch.sigmoid(m_logits)
        # preds = torch.where(preds < 0.5, 0, 1)
        # preds = preds.sum(1)
        # preds = torch.where(preds<2, 0, 1)

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def test_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
                        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
                        ) -> Tensor:
        m_logits, v_logits, a_logits, v_feats, a_feats = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_logits, a_logits, v_feats, a_feats, batch['v_label'], batch['a_label'],
                                               batch['c_label'], batch['m_label'])
        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        # emsemble
        # preds_a = torch.argmax(self.softmax(a_feats), dim=1)
        # preds_v = torch.argmax(self.softmax(v_feats), dim=1)
        # preds = preds_a + preds_v
        # preds = torch.where(preds<2, 0, 1)
        
        # multi-label
        # preds = torch.sigmoid(m_logits)
        # preds = torch.where(preds < 0.5, 0, 1)
        # preds = preds.sum(1)
        # preds = torch.where(preds<2, 0, 1)

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach(),
                "v_targets": batch['v_label'].detach(), "a_targets": batch['a_label'].detach(),
                'logits': m_logits.detach(), 'm_targets': batch['mm_label'].detach(), 'a_embeds': a_logits.detach(), 'v_embeds': v_logits.detach()}


    def training_step_end(self, training_step_outputs):
        # multi-class
        # train_acc = self.acc(training_step_outputs['preds'].clamp(0, 1), training_step_outputs['targets'].clamp(0, 1)).item()
        # train_auroc = self.auroc(training_step_outputs['preds'].clamp(0, 1), training_step_outputs['targets'].clamp(0, 1)).item()
        
        # others: common, ensemble, multi-label
        train_acc = self.acc(training_step_outputs['preds'], training_step_outputs['targets']).item()
        train_auroc = self.auroc(training_step_outputs['preds'], training_step_outputs['targets']).item()
        
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_auroc", train_auroc, prog_bar=True)

    def validation_step_end(self, validation_step_outputs):
        # multi-class
        # val_acc = self.acc(validation_step_outputs['preds'].clamp(0, 1), validation_step_outputs['targets'].clamp(0, 1)).item()
        # val_auroc = self.auroc(validation_step_outputs['preds'].clamp(0, 1), validation_step_outputs['targets'].clamp(0, 1)).item()
        
        # others: common, ensemble, multi-label
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

        # add for multi-class
        # preds = torch.clamp(preds, 0, 1)
        # targets = torch.clamp(targets, 0, 1)

        train_acc = self.acc(preds, targets).item()
        train_auroc = self.auroc(preds, targets).item()

        # train_real_f1score = self.f1score(preds, targets).item()
        # train_real_recall = self.recall(preds, targets).item()
        # train_real_precision = self.precision(preds, targets).item()
        #
        # train_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        # train_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        # train_fake_precision = self.precision(Opposite(preds), Opposite(targets)).item()

        print("Train - loss:", train_loss, "acc: ", train_acc, "auroc: ", train_auroc)
        # print("Train - loss:", train_loss, "acc: ", train_acc)

    def validation_epoch_end(self, validation_step_outputs):
        valid_loss = Average([i["loss"] for i in validation_step_outputs]).item()
        preds = [item for list in validation_step_outputs for item in list["preds"]]
        targets = [item for list in validation_step_outputs for item in list["targets"]]
        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        # add for multi-class
        # preds = torch.clamp(preds, 0, 1)
        # targets = torch.clamp(targets, 0, 1)

        # if valid_loss <= self.best_loss:
        self.best_acc = self.acc(preds, targets).item()
        self.best_auroc = self.auroc(preds, targets).item()
        self.best_real_f1score = self.f1score(preds, targets).item()
        self.best_real_recall = self.recall(preds, targets).item()
        self.best_real_precision = self.precisions(preds, targets).item()

        self.best_fake_f1score = self.f1score(Opposite(preds), Opposite(targets)).item()
        self.best_fake_recall = self.recall(Opposite(preds), Opposite(targets)).item()
        self.best_fake_precision = self.precisions(Opposite(preds), Opposite(targets)).item()

        self.best_loss = valid_loss
        # print("BEST Valid- loss: ", self.best_loss, "acc: ", self.best_acc)
        print("Valid loss: ", self.best_loss, "acc: ", self.best_acc, "auroc: ", self.best_auroc,
              "real_f1score:",
              self.best_real_f1score, "real_recall: ", self.best_real_recall, "real_precision: ",
              self.best_real_precision, "fake_f1score: ", self.best_fake_f1score, "fake_recall: ",
              self.best_fake_recall, "fake_precision: ", self.best_fake_precision)

    def test_epoch_end(self, test_step_outputs):
        test_loss = Average([i["loss"] for i in test_step_outputs]).item()
        preds = [item for list in test_step_outputs for item in list["preds"]]
        targets = [item for list in test_step_outputs for item in list["targets"]]
        m_targets = [item for list in test_step_outputs for item in list["m_targets"]]
        v_targets = [item for list in test_step_outputs for item in list["v_targets"]]
        a_targets = [item for list in test_step_outputs for item in list["a_targets"]]

        logits = [item for list in test_step_outputs for item in list["logits"]]
        a_embeds = [item for list in test_step_outputs for item in list["a_embeds"]]
        v_embeds = [item for list in test_step_outputs for item in list["v_embeds"]]

        logits = torch.stack(logits, dim=0)
        a_embeds = torch.stack(a_embeds, dim=0)
        v_embeds = torch.stack(v_embeds, dim=0)
        m_targets = torch.stack(m_targets, dim=0)
        v_targets = torch.stack(v_targets, dim=0)
        a_targets = torch.stack(a_targets, dim=0)

        preds = torch.stack(preds, dim=0)
        targets = torch.stack(targets, dim=0)

        v2_targets = v_targets + 2

        # tsne = TSNE(n_components=2, perplexity=4).fit_transform(logits.cpu().numpy())
        # visualize_tsne_2('test', tsne, m_targets.cpu().numpy())

        av_embeds = torch.cat([a_embeds, v_embeds], dim=0)
        av_targets = torch.cat([a_targets, v2_targets], dim=0)

        
        tsne = TSNE(n_components=2, perplexity=6).fit_transform(av_embeds.cpu().numpy())
        visualize_tsne_2('test', tsne, av_targets.cpu().numpy())

        # add for multi-class
        # preds = torch.clamp(preds, 0, 1)
        # targets = torch.clamp(targets, 0, 1)

        total_ff, total_fr, total_rf, total_rr = 0, 0, 0, 0
        sum_ff, sum_fr, sum_rf, sum_rr = 0, 0, 0, 0
        # sum_ff_f, sum_fr_f, sum_rf_f, sum_rr_f = 0, 0, 0, 0

        for i in range(preds.shape[0]):

            if (a_targets[i] == 1) and (v_targets[i] == 1):
                total_rr += 1
                if preds[i] == targets[i]:
                    sum_rr += 1
            if (a_targets[i] == 1) and (v_targets[i] == 0):
                total_rf += 1
                if preds[i] == targets[i]:
                    sum_rf += 1
            if (a_targets[i] == 0) and (v_targets[i] == 1):
                total_fr += 1
                if preds[i] == targets[i]:
                    sum_fr += 1
            if (a_targets[i] == 0) and (v_targets[i] == 0):
                total_ff += 1
                if preds[i] == targets[i]:
                    sum_ff += 1

        print(total_ff, total_fr, total_rf, total_rr, total_ff+total_fr+total_rf+total_rr)
        print(sum_ff, sum_fr, sum_rf, sum_rr, sum_ff+sum_fr+sum_rf+sum_rr)
        print(sum_ff/total_ff, sum_fr/total_fr, sum_rf/total_rf, sum_rr/total_rr, (sum_ff+sum_fr+sum_rf+sum_rr)/(total_ff+total_fr+total_rf+total_rr))

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
        # return {"loss": test_loss, "test_acc": test_acc}
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
