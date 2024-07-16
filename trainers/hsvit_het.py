import lightning as L
from lightning.fabric.utilities.throughput import measure_flops
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR
from torchmetrics.functional.classification import accuracy

from models.hsvit_het import HSViTHETForImageClassification


class HSViTHETModule(L.LightningModule):
    def __init__(
            self,
            optimizer_args: dict,
            label_smoothing: float,
            num_channels: int,
            image_size: int,
            sub_module_path: str,
            sub_module_feature_size: int,
            conv_kernel_num: int,
            enable_conv_layer: bool,
            attn_num: int,
            attn_depth: int,
            attn_embed_dim: int,
            num_heads: int,
            num_classes: int,
            dropout: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        # config for the optimizer and scheduler
        self.optimizer_args = optimizer_args
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.num_classes = num_classes
        self.model = HSViTHETForImageClassification(
            sub_module_path, sub_module_feature_size, conv_kernel_num, enable_conv_layer, attn_num, attn_depth,
            attn_embed_dim, num_heads, num_classes, dropout
        )
        # no training for pretrained Conv2d submodule
        for param in self.model.model.conv2d_submodule.parameters():
            param.requires_grad = False
        print(f"****** MODEL *******\n{self.model}\n********************")

        # FLOPS calculation (forward only)
        def sample_forward():
            # fake pixel_values: (batch_size, num_channels, height, width)
            pixel_values = torch.randn(
                size=(1, num_channels, image_size, image_size),
                device=self.device
            )
            return self.model(pixel_values)
        flops_per_batch = measure_flops(self.model, sample_forward)
        print(f"****** GFLOPs ******\n{flops_per_batch / 1e9}\n********************")

    def forward(self, images):
        logits = self.model(images)
        return logits

    def training_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "train")

    def validation_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "val")

    def test_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "test")

    def predict_step(self, batched_inputs, batch_idx):
        images, label = batched_inputs
        logits = self.model(images)
        print(f"Debug logits: {logits.shape}")

    def evaluate(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # measure accuracy and record loss
        acc1 = accuracy(logits, y, task="multiclass", num_classes=self.num_classes, top_k=1)
        acc5 = accuracy(logits, y, task="multiclass", num_classes=self.num_classes, top_k=5)

        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_acc1', acc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_acc5', acc5, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr_base = float(self.optimizer_args["lr_base"])
        lr_weight_decay = float(self.optimizer_args["lr_weight_decay"])
        lr_warmup_epochs = int(self.optimizer_args["lr_warmup_epochs"])

        optimizer = AdamW(self.parameters(), lr=lr_base, weight_decay=lr_weight_decay)
        steps_per_batch = int(self.trainer.estimated_stepping_batches / self.trainer.max_epochs)

        # linear warmup
        warmup_scheduler = \
            LambdaLR(optimizer, lr_lambda=lambda cur_step: cur_step / (steps_per_batch * lr_warmup_epochs))
        # cosine annealing lr scheduler
        cos_anneal_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches)
        schedulers = SequentialLR(optimizer, schedulers=[warmup_scheduler, cos_anneal_scheduler],
                                  milestones=[lr_warmup_epochs * steps_per_batch])
        return [optimizer], [{'scheduler': schedulers, 'interval': 'step'}]
