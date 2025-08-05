#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from PIL import Image

# --- 2. Definizione delle Classi per il Crop Intelligente ---
class ChestCropTransform:
    def __init__(self, final_size=224, crop_ratio=0.7):
        self.final_size = final_size
        self.crop_ratio = crop_ratio

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        elif torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = img
        if len(img_array.shape) == 3:
            img_array = img_array.squeeze()

        h, w = img_array.shape
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = img_array[start_h:start_h + crop_h, start_w:start_w + crop_w]
        return cropped

class AdaptiveChestCrop:
    def __init__(self, final_size=224, intensity_threshold=0.1):
        self.final_size = final_size
        self.intensity_threshold = intensity_threshold

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        elif torch.is_tensor(img):
            img_array = img.numpy()
        else:
            img_array = img
        if len(img_array.shape) == 3:
            img_array = img_array.squeeze()

        img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        grad_y = np.abs(np.diff(img_norm, axis=0))
        grad_x = np.abs(np.diff(img_norm, axis=1))
        edge_map = np.zeros_like(img_norm)
        edge_map[:-1, :] += grad_y
        edge_map[:, :-1] += grad_x
        coords = np.where(edge_map > self.intensity_threshold)

        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            h, w = img_array.shape
            padding = min(h, w) // 10
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            cropped = img_array[y_min:y_max, x_min:x_max]
        else:
            h, w = img_array.shape
            crop_h, crop_w = int(h * 0.7), int(w * 0.7)
            start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
            cropped = img_array[start_h:start_h + crop_h, start_w:start_w + crop_w]

        return cropped

# --- 3. Configurazione delle Trasformazioni ---
data_flag = 'chestmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# OPZIONE 1: Crop semplice (più conservativo)
data_transform_simple_crop = transforms.Compose([
    ChestCropTransform(final_size=224, crop_ratio=0.7),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # Direttamente da numpy a tensor
    transforms.Normalize(mean=[.5], std=[.5])
])

# OPZIONE 2: Crop adattivo (più aggressivo)
data_transform_adaptive_crop = transforms.Compose([
    AdaptiveChestCrop(final_size=224, intensity_threshold=0.05),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # Direttamente da numpy a tensor
    transforms.Normalize(mean=[.5], std=[.5])
])

# OPZIONE 3: Crop + Data Augmentation per migliorare la robustezza
data_transform_crop_augmented = transforms.Compose([
    ChestCropTransform(final_size=224, crop_ratio=0.7),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.Normalize(mean=[.5], std=[.5])
])

data_transform = data_transform_crop_augmented
print("Usando crop semplice (70% centrale)")

# --- 4. Caricamento Dataset ---
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"Dataset '{data_flag}' caricato.")
print(f"Dimensione training set: {len(train_dataset)}")
print(f"Dimensione test set: {len(test_dataset)}")

# --- 6. Modello Lineare ---
class SimpleLinearClassifier(nn.Module):
    def __init__(self):
        super(SimpleLinearClassifier, self).__init__()
        self.linear = nn.Linear(224 * 224, 1)

    def forward(self, x):
        x = x.view(-1, 224 * 224)
        return self.linear(x)

model = SimpleLinearClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modello su: {device}")

# --- 7. Configurazione Addestramento ---
num_positives = (train_dataset.labels[:, 1] == 1).sum().item()
num_negatives = (train_dataset.labels[:, 1] == 0).sum().item()
pos_weight = torch.tensor(num_negatives / num_positives, dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10

print(f"Pos weight: {pos_weight.item():.2f}")

# --- 8. Addestramento ---
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels[:, 1].float().to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoca [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Addestramento completato.")

# --- 9. Valutazione ---
model.eval()
correct = 0
total = 0
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels[:, 1].to(device).unsqueeze(1)
        outputs = model(images)
        predicted = torch.sigmoid(outputs).round().squeeze()
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()
        predicted_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy().squeeze()
        true_positives += ((predicted_np == 1) & (labels_np == 1)).sum()
        true_negatives += ((predicted_np == 0) & (labels_np == 0)).sum()
        false_positives += ((predicted_np == 1) & (labels_np == 0)).sum()
        false_negatives += ((predicted_np == 0) & (labels_np == 1)).sum()

accuracy = 100 * correct / total
sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

print(f"Accuracy sul test set: {accuracy:.2f}%")
print("\nMatrice di Confusione:")
print(f"TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")
print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}")

# --- 10. Estrazione pesi ---
trained_weights = model.linear.weight.data.cpu().numpy().squeeze()
trained_weights_reshaped = trained_weights.reshape((224, 224))

output_dir = 'data/images'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
plt.imshow(trained_weights_reshaped, cmap='RdBu_r', origin='upper')
plt.title('Pesi appresi (Cardiomegalia)', fontsize=14)
plt.colorbar(label='Peso')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.savefig(os.path.join(output_dir, 'pesi_con_crop.png'), dpi=300, bbox_inches='tight')
plt.show()

weights_dir = 'data/weights'
os.makedirs(weights_dir, exist_ok=True)
np.save(os.path.join(weights_dir, 'cardiomegaly_weights_224x224_trained_CROP.npy'), trained_weights_reshaped)
print(f"Pesi salvati in: {os.path.join(weights_dir, 'cardiomegaly_weights_224x224_trained_CROP.npy')}")

# --- 13. Salvataggio immagini per CUDA ---
print("\nPreparazione immagini di test per inferenza CUDA...")
all_test_images = np.array([np.array(img) for img, _ in test_dataset], dtype=np.float32)
print("Shape completo test set:", all_test_images.shape)

if all_test_images.ndim == 4 and all_test_images.shape[1] == 1:
    all_test_images = all_test_images.squeeze(1)

output_images_path = os.path.join(output_dir, 'test_images.npy')
np.save(output_images_path, all_test_images)
print(f"Tutte le immagini di test salvate in: {output_images_path}")

for i in range(5):
    single_img_path = os.path.join(output_dir, f'test_image_{i}.npy')
    np.save(single_img_path, all_test_images[i])
    print(f"Immagine singola salvata in: {single_img_path}")
