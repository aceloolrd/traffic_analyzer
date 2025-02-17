from model import PetModel
from config_loader import train_config 
from utils import vizualize_sample_data
from datamodule import SegmentationDataModule
from loss import BCEDiceLoss
import pytorch_lightning as pl

STAGE = None

dm = SegmentationDataModule()
dm.setup(stage='fit')

STEPS_PER_EPOCH = len(dm.train_dataloader())
CRITERION = BCEDiceLoss()

model = PetModel.load_from_checkpoint(
    f"{train_config["DEFAULT_ROOT_DIR"]}/checkpoints/{train_config["EXPERIMENT"]}/last.ckpt",
    arch=train_config["ARCH"],
    encoder_name=train_config["ENCODER_NAME"],
    encoder_weights=train_config["ENCODER_WEIGHTS"], 
    in_channels=train_config["IN_CHANELS"],
    out_classes=train_config["OUT_CLASSES"], 
    criterion=CRITERION,
    lr=train_config["LR"],
    batch_size=train_config["BATCH_SIZE"],
    epochs=train_config["EPOCHS"],
    steps_per_epoch=STEPS_PER_EPOCH
)

# model = PetModel.load_from_checkpoint(f"{train_config["DEFAULT_ROOT_DIR"]}/checkpoints/{train_config["EXPERIMENT"]}/last.ckpt")
# model.eval()

trainer = pl.Trainer(
    max_epochs=train_config["EPOCHS"],
    log_every_n_steps=1,
    # callbacks=callbacks,  
    accelerator=train_config["ACCELERATOR"],  
    devices=1,  
    enable_model_summary=True,  
    enable_checkpointing=True,  
    default_root_dir=train_config["DEFAULT_ROOT_DIR"],
    # logger=tensorboard_logger,
    fast_dev_run=False,  
    # fast_dev_run=True  
)

trainer.test(model, dm, ckpt_path=None)

# vizualize_sample_data(dm)