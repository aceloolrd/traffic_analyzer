from pathlib import Path
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Аугментации для изображений и масок
train_transform = A.Compose([
    A.Resize(256, 256),  # Изменение размера
    A.HorizontalFlip(p=0.5),  # Горизонтальный флип
    A.Normalize(mean=(0.5,), std=(0.5,)),  # Нормализация
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Фильтруем только изображения
        self.images = sorted([p.name for p in self.image_dir.glob("*.png")])  # Или "*.jpg"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.images[idx]
        mask_path = self.mask_dir / self.images[idx]  # Маска должна иметь то же имя

        # Загружаем изображение и маску с OpenCV (проверяем на `None`)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Ошибка загрузки: {img_path} или {mask_path}")

        mask = mask / 255.0  # Приводим к диапазону [0, 1]

        # Применяем аугментации
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return {"image": image, "mask": mask.unsqueeze(0)}  # Добавляем канал в маске [C=1]