"""
Create Kaggle Submission with Your Trained DINO Model
======================================================

This script uses your trained DINO model for feature extraction.

Usage:
    python create_submission_with_my_model.py \
        --data_dir ./data \
        --checkpoint ./checkpoints/best_model.pt \
        --output submission.csv \
        --k 5
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import argparse
import copy

# Import your DINO model classes
from dinov5_small import VisionTransformer, DINOHead, load_checkpoint
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ============================================================================
#                          MODEL CONFIGURATION
# ============================================================================
# IMPORTANT: These must match your training configuration exactly!
# If you trained with dinov5_small.py, use the SMALL config below
# If you trained with dinov5_full.py, use the FULL config below

image_size = 96

# SMALL MODEL CONFIG (for dinov5_small.py)
embed_dim = 256
patch_size = 8
num_heads = 8
mlp_dim = 1024
num_layers = 6
projection_dim = 65536

# FULL MODEL CONFIG (for dinov5_full.py) - uncomment if using full model
# embed_dim = 512
# patch_size = 4
# num_heads = 8
# mlp_dim = 2048
# num_layers = 12
# projection_dim = 65536


# ============================================================================
#                          FEATURE EXTRACTOR (Your Model)
# ============================================================================

class FeatureExtractor:
    """
    Feature extractor using your trained DINO model.
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize feature extractor with your trained model.
        
        Args:
            checkpoint_path: Path to your trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        print(f"Loading your trained DINO model from {checkpoint_path}...")
        self.device = device
        
        # Recreate model architecture (must match training config)
        student = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_classes=100,
            dropout=0.1
        ).to(device)
        
        teacher = copy.deepcopy(student)
        teacher.eval()
        
        # Projection heads
        student_head = DINOHead(in_dim=embed_dim, out_dim=projection_dim).to(device)
        teacher_head = DINOHead(in_dim=embed_dim, out_dim=projection_dim).to(device)
        
        # Optimizer (needed for checkpoint loading)
        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(student_head.parameters()),
            lr=1e-4,
            weight_decay=0.04,
            betas=(0.9, 0.999)
        )
        
        # Load checkpoint
        load_checkpoint(
            checkpoint_path,
            student, teacher,
            student_head, teacher_head,
            optimizer, device=device
        )
        
        # Use teacher model (EMA model) for feature extraction
        self.model = teacher
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Image transform (same as training evaluation)
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ Model loaded and frozen for evaluation")
        
    def extract_features(self, image):
        """
        Extract features from a single PIL Image.
        
        Args:
            image: PIL Image
        
        Returns:
            features: numpy array of shape (feature_dim,)
        """
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract CLS token features
        with torch.no_grad():
            feats = self.model(img_tensor, return_embedding=True)  # Returns CLS token
            feats = F.normalize(feats, dim=1)
        
        return feats.cpu().numpy()[0]
    
    def extract_batch_features(self, images):
        """
        Extract features from a batch of PIL Images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            features: numpy array of shape (batch_size, feature_dim)
        """
        # Transform images
        img_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Extract CLS token features
        with torch.no_grad():
            feats = self.model(img_tensors, return_embedding=True)  # Returns CLS token
            feats = F.normalize(feats, dim=1)
        
        return feats.cpu().numpy()


# ============================================================================
#                          DATA SECTION (Unchanged)
# ============================================================================

class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, image_list, labels=None, resolution=96):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
            resolution: Image resolution
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load and resize image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


def collate_fn(batch):
    """Custom collate function to handle PIL images"""
    if len(batch[0]) == 3:  # train/val (image, label, filename)
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:  # test (image, filename)
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames


# ============================================================================
#                          FEATURE EXTRACTION (Unchanged)
# ============================================================================

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train'):
    """
    Extract features from a dataloader.
    
    Args:
        feature_extractor: FeatureExtractor instance
        dataloader: DataLoader
        split_name: Name of split (for progress bar)
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: list of labels (or None for test)
        filenames: list of filenames
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:  # train/val
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:  # test
            images, filenames = batch
        
        # Extract features for batch
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    return features, labels, all_filenames


# ============================================================================
#                          KNN CLASSIFIER (Unchanged)
# ============================================================================

def train_knn_classifier(train_features, train_labels, val_features, val_labels, k=5):
    """
    Train KNN classifier on features.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        k: Number of neighbors
    
    Returns:
        classifier: Trained KNN classifier
    """
    print(f"\nTraining KNN classifier (k={k})...")
    
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',  # Weight by inverse distance
        metric='cosine',  # Cosine similarity for embeddings
        n_jobs=-1
    )
    
    classifier.fit(train_features, train_labels)
    
    # Evaluate
    train_acc = classifier.score(train_features, train_labels)
    val_acc = classifier.score(val_features, val_labels)
    
    print(f"\nKNN Results:")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return classifier


# ============================================================================
#                          SUBMISSION CREATION (Unchanged)
# ============================================================================

def create_submission(test_features, test_filenames, classifier, output_path):
    """
    Create submission.csv for Kaggle.
    
    Args:
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        classifier: Trained KNN classifier
        output_path: Path to save submission.csv
    """
    print("\nGenerating predictions on test set...")
    predictions = classifier.predict(test_features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission file created: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df['class_id'].max() <= 199, "Invalid class_id > 199"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")


# ============================================================================
#                          MAIN (Modified)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with Your Trained DINO Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to your trained model checkpoint (e.g., ./checkpoints/best_model.pt)')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--resolution', type=int, default=96,
                        help='Image resolution (96 for competition)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"\n✗ Error: Data directory '{data_dir}' does not exist!")
        print(f"\nPlease prepare the dataset first by running:")
        print(f"  python external_repos/fall2025_finalproject/testset_1/prepare_cub200_for_kaggle.py \\")
        print(f"      --download_dir ./raw_data --output_dir {data_dir}")
        return
    
    # Check for required CSV files
    required_files = ['train_labels.csv', 'val_labels.csv', 'test_images.csv']
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        print(f"\n✗ Error: Missing required files in '{data_dir}':")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease prepare the dataset first by running:")
        print(f"  python external_repos/fall2025_finalproject/testset_1/prepare_cub200_for_kaggle.py \\")
        print(f"      --download_dir ./raw_data --output_dir {data_dir}")
        return
    
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    print(f"\nCreating datasets (resolution={args.resolution}px)...")
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist(),
        resolution=args.resolution
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        labels=None,
        resolution=args.resolution
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize feature extractor with YOUR trained model
    feature_extractor = FeatureExtractor(checkpoint_path=args.checkpoint, device=device)
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, 'train'
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, 'val'
    )
    test_features, _, test_filenames = extract_features_from_dataloader(
        feature_extractor, test_loader, 'test'
    )
    
    # Train KNN classifier
    classifier = train_knn_classifier(
        train_features, train_labels,
        val_features, val_labels,
        k=args.k
    )
    
    # Create submission
    create_submission(test_features, test_filenames, classifier, args.output)
    
    print("\n" + "="*60)
    print("DONE! Now upload your submission.csv to Kaggle.")
    print("="*60)


if __name__ == "__main__":
    main()

