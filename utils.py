import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch

def denormalize(img):
    img = img * 0.5 + 0.5  # Обратная нормализация (если mean=0.5, std=0.5)
    return img.clamp(0, 1)  # Обрезаем в диапазон [0,1]

def vizualize_sample_data(dm):
    # Визуализируем примеры
    for dataset_name, dataset in zip(["Train", "Validation"], #, "Test"
                                    [dm.train_dataset, dm.val_dataset]): #, dm.test_dataset
        sample = dataset[0]
        
        plt.figure(figsize=(8, 4))
        
        # Восстанавливаем изображение в [0,1]
        img = denormalize(sample['image'])
        
        plt.subplot(1, 2, 1)
        plt.imshow(img.squeeze(0).numpy(), cmap="gray")  # Теперь диапазон [0,1]
        plt.title(f"{dataset_name} - Image")

        plt.subplot(1, 2, 2)
        plt.imshow(sample['mask'].squeeze(0).numpy(), cmap="gray")  # Преобразуем маску в uint8
        plt.title(f"{dataset_name} - Mask")

        plt.show()

import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, dataloader, num_images=3, save_dir="predictions"):
    """Визуализирует предсказания модели на нескольких изображениях."""
    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(dataloader):
        if i >= num_images:
            break

        image, mask = batch["image"], batch["mask"]
        with torch.no_grad():
            pred_mask = model(image).sigmoid().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image.squeeze().cpu().numpy(), cmap="gray")
        axes[0].set_title("Input Image")

        axes[1].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
        axes[1].set_title("Ground Truth")

        axes[2].imshow(pred_mask.squeeze(), cmap="gray")
        axes[2].set_title("Predicted Mask")

        plt.savefig(save_path / f"prediction_{i}.png")
        plt.close()

    print(f"Предсказания сохранены в {save_dir}")


def get_last_checkpoint(checkpoint_dir):
    """Возвращает путь к последнему чекпоинту, если он существует."""
    if checkpoint_dir is None:  # Если путь не указан в конфиге
        return None

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        return None  # Если папка пуста, возвращаем None
    
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        return None
    
    last_ckpt = max(ckpt_files, key=lambda f: f.stat().st_mtime)  # Берем последний по времени
    print(f"Найден чекпоинт: {last_ckpt}")
    return str(last_ckpt)
