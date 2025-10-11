import torch
import torch.nn as nn
import pytorch_lightning as pl
from .models import SmallUNet


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


class ArmorUNet(pl.LightningModule):
    """PyTorch Lightning module for armor plate detection"""

    def __init__(self, learning_rate=1e-4, weight_decay=1e-5, base_channels=32):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = SmallUNet(in_channels=3, out_channels=1, base_channels=base_channels)

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # For visualization
        self.example_input_array = torch.randn(1, 3, 640, 640)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_dice': dice}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)

        loss = self.criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)

        self.log('test_loss', loss)
        self.log('test_dice', dice)

        return {'test_loss': loss, 'test_dice': dice}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_dice',
                'interval': 'epoch',
                'frequency': 1
            }
        }
