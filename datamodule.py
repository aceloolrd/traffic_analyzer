from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import train_transform, val_transform, SegmentationDataset
from config_loader import train_config 
from pathlib import Path

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=train_config.get("DATASET_PATH", "./data/splits"), batch_size=train_config.get("BATCH_SIZE", 4), num_workers=train_config.get("NUM_WORKERS", 4)):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.current_train_batch_index = 0  
        
        # Проверяем, существует ли датасет
        if not Path(self.data_dir).exists():
            raise FileNotFoundError(f"Папка с датасетом {self.data_dir} не найдена.")

    def setup(self, stage=None):
        if stage == "fit":  # fit = train + val
            self.train_dataset = SegmentationDataset(
                image_dir=f"{self.data_dir}/train/image",
                mask_dir=f"{self.data_dir}/train/mask",
                transform=train_transform
            )
            self.val_dataset = SegmentationDataset(
                image_dir=f"{self.data_dir}/val/image",
                mask_dir=f"{self.data_dir}/val/mask",
                transform=val_transform
            )
            # print(f"Train size: {len(self.train_dataset)}")
            # print(f"Valid size: {len(self.val_dataset)}")

        if stage == "test":  # test = test dataset
            self.test_dataset = SegmentationDataset(
                image_dir=f"{self.data_dir}/test/image",
                mask_dir=f"{self.data_dir}/test/mask",
                transform=val_transform
            )
            # print(f"Test size: {len(self.test_dataset)}")

    def state_dict(self):
        return {"current_train_batch_index": self.current_train_batch_index}

    def load_state_dict(self, state_dict):
        self.current_train_batch_index = state_dict["current_train_batch_index"]

    def train_dataloader(self):
        if not hasattr(self, "train_dataset"):
            raise ValueError("train_dataset не инициализирован. Сначала вызовите `setup(stage='fit')`")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            raise ValueError("val_dataset не инициализирован. Сначала вызовите `setup(stage='fit')`")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            raise ValueError("test_dataset не инициализирован. Сначала вызовите `setup(stage='test')`")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True if self.num_workers > 0 else False)