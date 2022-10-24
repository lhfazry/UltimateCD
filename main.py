import argparse
from models.WaveCD import WaveCD
import pytorch_lightning as pl
from models.Exposure import Exposure
from datamodules.CDDataModule import CDDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="Train or test")
parser.add_argument("--model_name", type=str, default="wavevit", help="Model name")
parser.add_argument("--pretrained", type=str, default=None, help="File pretrained")
parser.add_argument("--data_dir", type=str, default="datasets/Exposure", help="Path ke datasets")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument("--accelerator", type=str, default='cpu', help="Accelerator")
parser.add_argument("--logs_dir", type=str, default='lightning_logs', help="Log dir")
parser.add_argument("--variant", type=str, default='base', help="Variant model")
parser.add_argument("--multi_stage_training", action='store_true', help="Multi stage training")
parser.add_argument("--log", action='store_true', help="log")

params = parser.parse_args()

if __name__ == '__main__':
    mode = params.mode
    data_dir = params.data_dir
    model_name = params.model_name
    pretrained = params.pretrained
    batch_size = params.batch_size
    ckpt_path = params.ckpt_path
    max_epochs = params.max_epochs
    num_workers = params.num_workers
    accelerator = params.accelerator
    logs_dir = params.logs_dir
    log = params.log
    variant = params.variant

    logger = TensorBoardLogger(save_dir=logs_dir, name=model_name)

    data_module = CDDataModule(
                    root_data_path = data_dir,
                    pre_image_dir = "A",
                    post_image_dir = "B",
                    mask_image_dir = "label",
                    batch_size=batch_size,
                    num_workers=num_workers)

    if variant == 'small':
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
        embed_dim = 96
    else: # base
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
        embed_dim = 128 

    wavecd = WaveCD(embed_dim=embed_dim, 
                    depths=depths, 
                    num_heads=num_heads, 
                    batch_size=batch_size, 
                    wavevit_checkpoint=pretrained)

    trainer = pl.Trainer(accelerator=accelerator, 
                max_epochs=max_epochs, 
                num_sanity_val_steps=1, 
                auto_scale_batch_size=True, 
                enable_model_summary=True,
                logger=logger,
                precision=16,
                log_every_n_steps=40,
                callbacks=[EarlyStopping(monitor="val_loss")])

    if mode == 'train':
        trainer.fit(model=wavecd, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'validate':
        if not log:
            trainer.logger = False

        trainer.validate(model=wavecd, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'test':
        if not log:
            trainer.logger = False

        trainer.test(model=wavecd, datamodule=data_module, ckpt_path=ckpt_path)

    if mode == 'predict':
        if not log:
            trainer.logger = False

        predicts = trainer.predict(model=wavecd, datamodule=data_module, ckpt_path=ckpt_path)

        for predict in predicts:
            print(predict)
            print('\n')
