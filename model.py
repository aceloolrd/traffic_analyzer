import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

class PetModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, criterion, lr, batch_size, epochs, steps_per_epoch, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.model = smp.create_model(
            arch=arch, 
            encoder_name=encoder_name, 
            encoder_weights=encoder_weights, 
            in_channels=in_channels, 
            classes=out_classes, 
            **kwargs
        )
        
        self.loss_fn = criterion

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch["image"], batch["mask"].float()  # Маску преобразуем в float
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = torch.where(prob_mask > 0.5, 1, 0)

        # IoU метрики
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Логирование
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size) #, reduce_fx="mean"
        self.log(f"{stage}_per_image_iou", per_image_iou, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log(f"{stage}_dataset_iou", dataset_iou, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                steps_per_epoch=self.hparams.steps_per_epoch,
                epochs=self.hparams.epochs,
                # pct_start=0.3,
                # anneal_strategy="cos",
                # div_factor=10,
                # final_div_factor=1000,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}