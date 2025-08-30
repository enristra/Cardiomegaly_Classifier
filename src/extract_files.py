import medmnist
from medmnist import ChestMNIST
import numpy as np
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


root = "/scratch.hpc/enrico.strangio/medmnist_cache"
os.makedirs(root, exist_ok=True)

split= 'train'
dataset = ChestMNIST(root=root, split = split, download=True, size=224)
num_samples=20000

out_dir = "/scratch.hpc/enrico.strangio/Cardiomegaly_Classifier/data"
images_dir= f"{out_dir}/images"
labels_path= f"{out_dir}/labels.npy"
images = dataset.imgs
labels = dataset.labels

images_subset = images[:num_samples]
labels_subset = labels[:num_samples]

os.makedirs(images_dir, exist_ok=True)
for i, img in enumerate(images_subset):
    np.save(os.path.join(images_dir, f"image_{i:06d}.npy"), img)

# === SAVE LABELS ===
np.save(labels_path, labels_subset)

print(f"Saved {num_samples} images to {images_dir}")
print(f"Saved labels to {labels_path} with shape {labels_subset.shape}")