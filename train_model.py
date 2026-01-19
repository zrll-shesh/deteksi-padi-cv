import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
import time
from collections import defaultdict
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ==================== CONFIGURATION ====================
class Config:
    DATA_DIR = 'rice leaf diseases dataset'
    CLASSES = ['Bacterialblight', 'Brownspot', 'Leafsmut']
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 12
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cpu')
    NUM_WORKERS = 2
    SAMPLE_RATIO = 0.25
    MODEL_SAVE_PATH = 'models'
    RESULTS_DIR = 'results'
    RANDOM_SEED = 42

# Set random seeds
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# ==================== DATASET PREPARATION ====================
class RiceLeafDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_dataset_paths(data_dir, sample_ratio=0.25):
    """Load image paths and labels with sampling"""
    image_paths = []
    labels = []
    class_counts = defaultdict(int)
    
    for idx, class_name in enumerate(Config.CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found!")
            continue
            
        files = [f for f in os.listdir(class_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sample images
        sample_size = int(len(files) * sample_ratio)
        sampled_files = np.random.choice(files, size=sample_size, replace=False)
        
        for filename in sampled_files:
            img_path = os.path.join(class_dir, filename)
            image_paths.append(img_path)
            labels.append(idx)
            class_counts[class_name] += 1
    
    print("\n=== Dataset Statistics ===")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    print(f"Total images: {len(image_paths)}")
    
    return image_paths, labels, class_counts

# ==================== DATA AUGMENTATION ====================
def get_train_transforms():
    """Advanced augmentation for training"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])

def get_val_transforms():
    """Standard transforms for validation/test"""
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ==================== MODEL ARCHITECTURE ====================
class ResNet34RiceClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet34RiceClassifier, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_training_history(history, save_path):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training history plot to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=Config.CLASSES,
        yticklabels=Config.CLASSES,
        cbar_kws={'label': 'Count'},
        linewidths=1,
        linecolor='gray'
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_normalized_confusion_matrix(y_true, y_pred, save_path):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
        xticklabels=Config.CLASSES,
        yticklabels=Config.CLASSES,
        cbar_kws={'label': 'Percentage'},
        linewidths=1,
        linecolor='gray',
        vmin=0, vmax=1
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved normalized confusion matrix to {save_path}")

def plot_class_performance(y_true, y_pred, save_path):
    """Plot per-class performance metrics"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics_df = pd.DataFrame({
        'Class': Config.CLASSES,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision
    axes[0, 0].bar(metrics_df['Class'], metrics_df['Precision'], 
                   color='skyblue', edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Precision by Class', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[0, 1].bar(metrics_df['Class'], metrics_df['Recall'], 
                   color='lightcoral', edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Recall', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Recall by Class', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # F1-Score
    axes[1, 0].bar(metrics_df['Class'], metrics_df['F1-Score'], 
                   color='lightgreen', edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Support
    axes[1, 1].bar(metrics_df['Class'], metrics_df['Support'], 
                   color='wheat', edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Support (samples)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Support by Class', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class performance plot to {save_path}")

def plot_roc_curves(y_true, y_probs, save_path):
    """Plot ROC curves for each class"""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = len(Config.CLASSES)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for i, color in enumerate(colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{Config.CLASSES[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Multi-class', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {save_path}")

def plot_dataset_distribution(class_counts, save_path):
    """Plot dataset class distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    # Bar plot
    axes[0].bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_title('Dataset Distribution - Bar Chart', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=15)
    
    for i, v in enumerate(counts):
        axes[0].text(i, v + max(counts)*0.02, str(v), 
                    ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts, labels=classes, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 11})
    axes[1].set_title('Dataset Distribution - Pie Chart', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dataset distribution plot to {save_path}")

def plot_sample_images(image_paths, labels, save_path, samples_per_class=3):
    """Plot sample images from dataset"""
    fig, axes = plt.subplots(len(Config.CLASSES), samples_per_class, 
                            figsize=(12, 10))
    
    for class_idx, class_name in enumerate(Config.CLASSES):
        class_images = [img for img, lbl in zip(image_paths, labels) 
                       if lbl == class_idx]
        selected = np.random.choice(class_images, samples_per_class, replace=False)
        
        for img_idx, img_path in enumerate(selected):
            img = Image.open(img_path)
            axes[class_idx, img_idx].imshow(img)
            axes[class_idx, img_idx].axis('off')
            
            if img_idx == 0:
                axes[class_idx, img_idx].set_ylabel(
                    class_name, fontsize=11, fontweight='bold', rotation=0,
                    ha='right', va='center'
                )
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample images to {save_path}")

# ==================== MAIN TRAINING PIPELINE ====================
def main():
    print("\n" + "="*60)
    print("AgroSight - Rice Disease Detection Training Pipeline")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels, class_counts = load_dataset_paths(
        Config.DATA_DIR, Config.SAMPLE_RATIO
    )
    
    # Plot dataset distribution
    plot_dataset_distribution(
        class_counts,
        os.path.join(Config.RESULTS_DIR, 'dataset_distribution.png')
    )
    
    # Plot sample images
    plot_sample_images(
        image_paths, labels,
        os.path.join(Config.RESULTS_DIR, 'sample_images.png')
    )
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, 
        random_state=Config.RANDOM_SEED, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=Config.RANDOM_SEED, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"Training: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Testing: {len(X_test)} images")
    
    # Create datasets and dataloaders
    train_dataset = RiceLeafDataset(X_train, y_train, get_train_transforms())
    val_dataset = RiceLeafDataset(X_val, y_val, get_val_transforms())
    test_dataset = RiceLeafDataset(X_test, y_test, get_val_transforms())
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS
    )
    
    # Initialize model
    print("\nInitializing ResNet-34 model...")
    model = ResNet34RiceClassifier(num_classes=len(Config.CLASSES))
    model = model.to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    print("="*60)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        val_loss, val_acc, _, _, _ = validate_epoch(
            model, val_loader, criterion, Config.DEVICE
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] ({epoch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
            )
            print(f"  >>> New best model saved! (Val Acc: {val_acc:.4f})")
        print("-"*60)
    
    # Plot training history
    plot_training_history(
        history,
        os.path.join(Config.RESULTS_DIR, 'training_history.png')
    )
    
    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(
        torch.load(os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth'))
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate_epoch(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Generate visualizations
    print("\nGenerating evaluation visualizations...")
    
    plot_confusion_matrix(
        test_labels, test_preds,
        os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
    )
    
    plot_normalized_confusion_matrix(
        test_labels, test_preds,
        os.path.join(Config.RESULTS_DIR, 'confusion_matrix_normalized.png')
    )
    
    plot_class_performance(
        test_labels, test_preds,
        os.path.join(Config.RESULTS_DIR, 'class_performance.png')
    )
    
    plot_roc_curves(
        test_labels, test_probs,
        os.path.join(Config.RESULTS_DIR, 'roc_curves.png')
    )
    
    # Classification report
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(
        test_labels, test_preds,
        target_names=Config.CLASSES,
        digits=4
    ))
    
    # Save metrics to JSON
    metrics = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'classification_report': classification_report(
            test_labels, test_preds,
            target_names=Config.CLASSES,
            output_dict=True
        )
    }
    
    with open(os.path.join(Config.RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Best model saved to: {Config.MODEL_SAVE_PATH}/best_model.pth")
    print(f"Results saved to: {Config.RESULTS_DIR}/")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()