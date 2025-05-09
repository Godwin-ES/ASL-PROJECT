import torch
import pytorch_lightning as pl
import timm
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score

class ASLModel(pl.LightningModule):
    def __init__(self, num_classes=26, model_name='efficientnet_b0', lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained EfficientNet
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.train_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.train_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')


    def forward(self, x):
        return self.backbone(x)

    def compute_metrics(self, logits, y, mode="train"):
        preds = logits.softmax(dim=-1)
        acc = self.train_acc if mode == "train" else self.val_acc
        precision = self.train_precision if mode == "train" else self.val_precision
        recall = self.train_recall if mode == "train" else self.val_recall
        f1 = self.train_f1 if mode == "train" else self.val_f1
        return {
            "acc": acc(preds, y),
            "precision": precision(preds, y),
            "recall": recall(preds, y),
            "f1": f1(preds, y)
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        metrics = self.compute_metrics(logits, y, mode="train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", metrics["acc"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_precision", metrics["precision"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recall", metrics["recall"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", metrics["f1"], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        metrics = self.compute_metrics(logits, y, mode="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", metrics["acc"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", metrics["precision"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", metrics["recall"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", metrics["f1"], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
