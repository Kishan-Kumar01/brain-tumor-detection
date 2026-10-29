"""
BrainSpark Laboratory - Model Training Engine (Quick 1-Epoch Mode)
Architecture: AT-Net (Hybrid Swin-Transformer + Attention U-Net)
Dataset Structure: Training & Testing folders with 4 sub-classes
Author: Kishan Kumar (24SCSE1430269) & Team
"""

import os
import zipfile
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Import model architecture from engine.py
from engine import ATNetDualHead, zscore_normalize

# ==========================================
# 1. ZIP EXTRACTION ENGINE
# ==========================================

DATASET_ZIP_PATH = "brain_tumor_dataset.zip"
EXTRACT_DIR = "./data_extracted"


def prepare_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        if os.path.exists(zip_path):
            print(f"[INFO] Extracting {zip_path} to {extract_to}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            print("[SUCCESS] Dataset extraction completed successfully.")
        else:
            raise FileNotFoundError(
                f"[ERROR] Could not find '{zip_path}' in project directory."
            )
    else:
        print("[INFO] Dataset directory already exists. Skipping extraction.")


# ==========================================
# 2. DATASET CLASS & PARSER
# ==========================================


class KaggleBrainTumorDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load & Preprocess
        image = Image.open(img_path).convert("L")
        img_np = np.array(image.resize((256, 256)), dtype=np.float32)
        norm_img = zscore_normalize(img_np)

        tensor_img = torch.tensor(norm_img).unsqueeze(0)  # [1, 256, 256]
        tensor_label = torch.tensor(label, dtype=torch.long)

        return tensor_img, tensor_label


def load_data_from_split(base_dir, split_type="Training"):
    """Loads file paths and labels specifically for Training or Testing folder"""
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    file_paths = []
    labels = []

    for cls_idx, cls_name in enumerate(classes):
        # Searching case-insensitive matching paths
        search_path = os.path.join(base_dir, "**", split_type, "**", cls_name, "*.*")
        found_files = glob.glob(search_path, recursive=True)

        # Fallback for uppercase folder names (e.g., Glioma / Training)
        if not found_files:
            search_path = os.path.join(
                base_dir, "**", split_type, "**", cls_name.capitalize(), "*.*"
            )
            found_files = glob.glob(search_path, recursive=True)

        # Filter valid image extensions
        found_files = [
            f for f in found_files if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for f in found_files:
            file_paths.append(f)
            labels.append(cls_idx)

    print(f"[DATASET INFO] {split_type} Set: Found {len(file_paths)} images.")
    return file_paths, labels


# ==========================================
# 3. TRAINING ENGINE
# ==========================================


def train_model():
    # Step A: Extract Zip (Skips if already extracted)
    prepare_dataset(DATASET_ZIP_PATH, EXTRACT_DIR)

    # Step B: Load Pre-split Training & Testing Data
    train_files, train_labels = load_data_from_split(EXTRACT_DIR, split_type="Training")
    val_files, val_labels = load_data_from_split(EXTRACT_DIR, split_type="Testing")

    if len(train_files) == 0:
        # Fallback if folder casing is lowercase 'training'
        train_files, train_labels = load_data_from_split(
            EXTRACT_DIR, split_type="training"
        )
        val_files, val_labels = load_data_from_split(EXTRACT_DIR, split_type="testing")

    if len(train_files) == 0:
        raise ValueError(
            "[ERROR] No images found. Please verify folder names inside zip."
        )

    train_dataset = KaggleBrainTumorDataset(train_files, train_labels)
    val_dataset = KaggleBrainTumorDataset(val_files, val_labels)

    # Batch Size increased to 32 for faster CPU execution
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Step C: Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training AT-Net on: {device}")

    model = ATNetDualHead(in_channels=1, num_classes_seg=4, num_classes_cls=4).to(
        device
    )

    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # EPOCHS SET TO 1 FOR QUICK EXECUTION
    epochs = 1
    best_val_acc = 0.0

    print("\n--- STARTING AT-NET TRAINING LOOP (1 EPOCH) ---")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]"
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            cls_logits, _ = model(images)
            loss = criterion_cls(cls_logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(cls_logits, dim=1)
            train_correct += torch.sum(preds == labels).item()

        train_acc = train_correct / len(train_dataset)

        # Validation Step
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                cls_logits, _ = model(images)
                preds = torch.argmax(cls_logits, dim=1)
                val_correct += torch.sum(preds == labels).item()

        val_acc = val_correct / len(val_dataset)

        print(
            f"\nEpoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss/len(train_dataset):.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

        # Save Checkpoint
        torch.save(model.state_dict(), "best_atnet_weights.pth")
        print(
            f"[CHECKPOINT] Model weights saved successfully to 'best_atnet_weights.pth'!"
        )

    print("\n[COMPLETE] AT-Net 1-Epoch Training Completed Successfully!")


if __name__ == "__main__":
    train_model()
