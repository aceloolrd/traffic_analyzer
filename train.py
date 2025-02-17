from pytorch_lightning.loggers import TensorBoardLogger
from config_loader import train_config 
from callbacks import callbacks
from datamodule import SegmentationDataModule
from model import PetModel
import pytorch_lightning as pl
from loss import BCEDiceLoss
from pathlib import Path
import os
from utils import get_last_checkpoint

if __name__ == "__main__":
    
    os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
    
    STAGE = "fit"

    dm = SegmentationDataModule()
    dm.setup(stage=STAGE)

    STEPS_PER_EPOCH = len(dm.train_dataloader())
    CRITERION = BCEDiceLoss()

    model = PetModel(
        arch=train_config.get("ARCH", "unetplusplus"),
        encoder_name=train_config.get("ENCODER_NAME", "resnext50_32x4d"),
        encoder_weights=train_config.get("ENCODER_WEIGHTS", "ssl"),
        in_channels=train_config.get("IN_CHANNELS", 1),
        out_classes=train_config.get("OUT_CLASSES", 1),
        criterion=CRITERION,
        lr=train_config.get("LR", 0.001),
        batch_size=train_config.get("BATCH_SIZE", 4),
        epochs=train_config.get("EPOCHS", 50),
        steps_per_epoch=STEPS_PER_EPOCH
    )

    log_dir = Path(__file__).parent / train_config["DEFAULT_ROOT_DIR"] / "tb_logs"
    tensorboard_logger = TensorBoardLogger(str(log_dir), name=train_config["EXPERIMENT"])

    trainer = pl.Trainer(
        max_epochs=train_config.get("EPOCHS", 50),
        log_every_n_steps=1,
        callbacks=callbacks,  
        accelerator=train_config.get("ACCELERATOR", "gpu"),
        devices=1,  
        enable_model_summary=True,  
        enable_checkpointing=True,  
        default_root_dir=train_config.get("DEFAULT_ROOT_DIR", "experiments"),
        logger=tensorboard_logger,
        # num_sanity_val_steps=0,  # Отключаем sanity check
        fast_dev_run=False,  
        # fast_dev_run=True  
    )

    # Проверяем путь к чекпоинту
    ckpt_path = Path(__file__).parent / train_config.get("CKPT_PATH", None)  # Если ключ отсутствует, будет None
    if isinstance(ckpt_path, str) and ckpt_path.lower() == "none":
        ckpt_path = None  # Преобразуем "None" (строку) в None (объект)
    
    # ckpt_path = get_last_checkpoint(Path(__file__).parent / train_config.get("CKPT_PATH", None))
    
    trainer.fit(model, dm, ckpt_path=ckpt_path)


