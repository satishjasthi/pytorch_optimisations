# train.py

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import PyTorchProfiler
from torch import nn
import segmentation_models_pytorch as smp
import torch
from data import get_dataloaders

class SegmentationModule(pl.LightningModule):
    def __init__(self, num_classes=18, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        val_loss = self.loss_fn(logits, masks)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main():
    pl.seed_everything(42)
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=16, img_size=(256,256))
    model = SegmentationModule(num_classes=18, lr=1e-3)

    profiler = PyTorchProfiler(
        dirpath=".",
        filename="pytorch_profile.txt",
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        export_to_chrome=True,
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        profiler=profiler,
    )

    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, test_dl)

if __name__ == "__main__":
    main()