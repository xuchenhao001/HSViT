import lightning as L
from lightning.fabric.utilities.throughput import measure_flops
import torch
from torchmetrics.functional.classification import accuracy
from transformers import AutoConfig, AutoModelForImageClassification


class SOTAModule(L.LightningModule):
    def __init__(
            self,
            optimizer_args: dict,
            num_channels: int,
            image_size: int,
            sota_config_path: str,
            num_classes: int
    ):
        super().__init__()
        self.save_hyperparameters()
        # config for the optimizer and scheduler
        self.optimizer_args = optimizer_args
        self.criterion = torch.nn.CrossEntropyLoss()

        self.num_classes = num_classes
        sota_config = AutoConfig.from_pretrained(sota_config_path)
        self.model = AutoModelForImageClassification.from_config(sota_config)
        print(f"****** MODEL *******\n{self.model}\n********************")

        # FLOPS calculation (forward only)
        def sample_forward():
            # fake pixel_values: (batch_size, num_channels, height, width)
            pixel_values = torch.randn(
                size=(1, num_channels, image_size, image_size),
                device=self.device
            )
            return self.model(pixel_values).logits
        flops_per_batch = measure_flops(self.model, sample_forward)
        print(f"****** GFLOPs ******\n{flops_per_batch / 1e9}\n********************")

    def forward(self, images):
        logits = self.model(images).logits
        return logits

    def training_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "train")

    def validation_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "val")

    def test_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "test")

    def predict_step(self, batched_inputs, batch_idx):
        images, label = batched_inputs
        logits = self.model(images).logits
        print(f"Debug logits: {logits.shape}")

    def evaluate(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # measure accuracy and record loss
        acc1 = accuracy(logits, y, task="multiclass", num_classes=self.num_classes, top_k=1)
        acc5 = accuracy(logits, y, task="multiclass", num_classes=self.num_classes, top_k=5)

        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log(f'{stage}_acc1', acc1, on_step=True, on_epoch=True, logger=True)
        self.log(f'{stage}_acc5', acc5, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        lr_base = float(self.optimizer_args["lr_base"])
        lr_weight_decay = float(self.optimizer_args["lr_weight_decay"])

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr_base, weight_decay=lr_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

