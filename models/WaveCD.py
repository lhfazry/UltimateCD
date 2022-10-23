from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

from torch.nn import functional as F
from backbones.wavevit import WaveViT
from einops import rearrange
from utils.tensor_utils import Reduce
from sklearn.metrics import r2_score


class WaveCD(pl.LightningModule):
    def __init__(self,
                pretrained=None,
                in_chans=3, 
                stem_hidden_dim = 32,
                embed_dims=[64, 128, 320, 512],
                depths=[3, 4, 12, 3], 
                num_heads=[3, 6, 12, 24],
                mlp_ratios=[8, 8, 4, 4], 
                wavevit_checkpoint=None, 
                batch_size=8):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.wavevit = WaveViT(
            in_chans=in_chans, 
            stem_hidden_dim = stem_hidden_dim,
            embed_dims=embed_dims,
            num_heads=num_heads, 
            mlp_ratios=mlp_ratios, 
            depths=depths, 
            token_label=True
        )
        
        

    def forward_features(self, x):
        

        return x

    def forward_head(self, x):
        

        return x

    def forward(self, x):
        x = self.forward_features(x) 
        x = self.forward_head(x)
        return x

    def training_step(self, batch, batch_idx):
        prediction_label = self(batch['video'])
        loss = self.loss_fn(prediction_label, batch['label'])
        acc = self.accuracy((prediction_label.sigmoid() > 0.5).long(), batch['label'])

        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.log("acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'train': loss}, global_step=self.current_epoch) 
        self.logger.experiment.add_scalars('acc', {'train': acc}, global_step=self.current_epoch) 

        return loss

    def validation_step(self, batch, batch_idx):
        prediction_label = self(batch['video'])
        loss = self.loss_fn(prediction_label, batch['label'])
        acc = self.accuracy((prediction_label.sigmoid() > 0.5).long(), batch['label'])

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.log("val_acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'val': loss}, global_step=self.current_epoch) 
        self.logger.experiment.add_scalars('acc', {'val': acc}, global_step=self.current_epoch) 
        
    def test_step(self, batch, batch_idx):
        prediction_label = self(batch['video'])
        pred_label_sigmoid = prediction_label.sigmoid()
        #print(f"predict: {prediction_label}")
        #print(f"label: {batch['label']}")

        loss = self.loss_fn(prediction_label, batch['label'])
        cm = self.confusion_matrix(pred_label_sigmoid, batch['label'].long())
        self.accuracy(pred_label_sigmoid, batch['label'])
        self.f1_score(pred_label_sigmoid, batch['label'])
        self.prec(pred_label_sigmoid, batch['label'])
        self.recall(pred_label_sigmoid, batch['label'])

        cm_mean = cm.float().mean(0)
        
        #true negatives for class i in M(0,0)
        #false positives for class i in M(0,1)
        #false negatives for class i in M(1,0)
        #true positives for class i in M(1,1)

        self.log('test_loss', loss, on_epoch=True)
        self.log('accuracy', self.accuracy, on_epoch=True)

        self.log('TN', cm_mean[0,0], on_epoch=True)
        self.log('FP', cm_mean[0,1], on_epoch=True)
        self.log('FN', cm_mean[1,0], on_epoch=True)
        self.log('TP', cm_mean[1,1], on_epoch=True)

        self.log('precision', self.prec, on_epoch=True)
        self.log('recall', self.recall, on_epoch=True)
        self.log('f1_score', self.f1_score, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        return self.shared_step(batch, 'predict')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        #optimizer = torch.optim.AdamW(self.parameters())
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)

        return [optimizer]#, [lr_scheduler]