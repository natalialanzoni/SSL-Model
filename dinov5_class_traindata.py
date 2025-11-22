# DINO Self-Supervised Learning Implementation
# Using competition pretraining dataset from HuggingFace

# Note: Install dependencies with: pip install faiss-cpu torch torchvision datasets

# pip install faiss-cpu
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import faiss
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import os
import sys
import matplotlib.pyplot as plt  # For plotting loss and k-NN accuracy
import copy
import zipfile
import glob
from huggingface_hub import snapshot_download

class DINOTarget:
    def __init__(self, dim, momentum=0.9, teacher_temp=0.07, device="cuda"):
        # DINO paper uses teacher_temp=0.07 (not 0.04)
        self.center = torch.zeros(1, dim, device=device)
        self.momentum = momentum
        self.teacher_temp = teacher_temp

    def __call__(self, teacher_logits):
        # center
        t = teacher_logits - self.center
        # sharpen
        t = t / self.teacher_temp
        t = F.softmax(t, dim=-1)
        # update center
        self.center = self.center * self.momentum + (1 - self.momentum) * teacher_logits.mean(dim=0, keepdim=True)
        return t.detach()

def load_checkpoint(checkpoint_path, student, teacher, student_head, teacher_head, optimizer=None, device="cuda"):
    """Load a checkpoint and restore model states"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(checkpoint['student_state_dict'])
    teacher.load_state_dict(checkpoint['teacher_state_dict'])
    student_head.load_state_dict(checkpoint['student_head_state_dict'])
    teacher_head.load_state_dict(checkpoint['teacher_head_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, k-NN acc: {checkpoint.get('knn_acc', 'N/A')}")
    return checkpoint['epoch'], checkpoint.get('knn_acc', 0.0)

def update_teacher(student, teacher, student_head, teacher_head, ema_m):
    # Update backbone parameters
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data = ema_m * t_param.data + (1 - ema_m) * s_param.data
    # Update projection head parameters
    for s_param, t_param in zip(student_head.parameters(), teacher_head.parameters()):
        t_param.data = ema_m * t_param.data + (1 - ema_m) * s_param.data

def dino_loss(student_logits, teacher_probs, student_temp=0.1):
    # student_logits: raw output of student projection head (NOT normalized)
    # teacher_probs: output of teacher after centering and sharpening
    # DINO paper uses student_temp=0.1 (which we have)
    # Apply temperature scaling to student logits before log_softmax
    student_log_probs = F.log_softmax(student_logits / student_temp, dim=-1)
    loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
    return loss

def koleo_loss(embeddings, k=3, eps=1e-8):
    """
    KoLeo (Kozachenko-Leonenko) regularization loss.
    Encourages diverse representations by maximizing entropy using k-NN distances.
    
    Args:
        embeddings: Tensor of shape (batch_size, embed_dim) - normalized embeddings
        k: Number of nearest neighbors to use
        eps: Small epsilon for numerical stability
    
    Returns:
        KoLeo loss (negative entropy, so we minimize it to maximize entropy)
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise distances
    # embeddings: (B, D), compute (B, B) distance matrix
    dot_product = torch.mm(embeddings, embeddings.t())  # (B, B)
    # For normalized vectors, dot product = cosine similarity
    # Distance = 1 - cosine_similarity (for normalized vectors)
    distances = 1 - dot_product  # (B, B)
    
    # Set diagonal to large value (self-distance should be ignored)
    distances.fill_diagonal_(float('inf'))
    
    # Find k nearest neighbors for each sample
    # Get k smallest distances (excluding self)
    knn_distances, _ = torch.topk(distances, k=k, dim=1, largest=False)  # (B, k)
    
    # KoLeo entropy estimate: -log(knn_distance) averaged
    # We want to maximize entropy, so we minimize negative entropy
    # Add eps to avoid log(0)
    log_distances = torch.log(knn_distances + eps)  # (B, k)
    koleo = -log_distances.mean()  # Negative entropy (we minimize this)
    
    return koleo

def train_dino(train_loader, student, teacher, student_head, teacher_head,
               optimizer, device="cuda", num_epochs=50, ema_m=0.996, knn_eval_freq=5,
               warmup_epochs=10, save_dir="./checkpoints", save_freq=10, koleo_weight=0.1,
               knn_train_loader=None, knn_test_loader=None, resume_from=None):

    # Create checkpoint directory
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize or resume from checkpoint
    start_epoch = 0
    best_knn_acc = 0.0
    losses = []
    knn_accuracies = []
    epochs_evaluated = []
    scaler = torch.amp.GradScaler()
    
    # Check for resume checkpoint
    resume_checkpoint_path = None
    if resume_from:
        resume_checkpoint_path = resume_from
    elif save_dir:
        # Check for latest checkpoint
        latest_checkpoint = Path(save_dir) / "checkpoint_latest.pt"
        if latest_checkpoint.exists():
            resume_checkpoint_path = str(latest_checkpoint)
    
    if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
        print(f"\n🔄 Resuming training from {resume_checkpoint_path}...")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint['student_state_dict'])
        teacher.load_state_dict(checkpoint['teacher_state_dict'])
        student_head.load_state_dict(checkpoint['student_head_state_dict'])
        teacher_head.load_state_dict(checkpoint['teacher_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_knn_acc = checkpoint.get('best_knn_acc', 0.0)
        losses = checkpoint.get('losses', [])
        knn_accuracies = checkpoint.get('knn_accuracies', [])
        epochs_evaluated = checkpoint.get('epochs_evaluated', [])
        print(f"✓ Resumed from epoch {start_epoch}, best k-NN acc: {best_knn_acc:.2f}%")
    else:
        # Initialize teacher as a copy of student
        teacher.load_state_dict(student.state_dict())
        teacher.eval()
        # Initialize teacher_head as a copy of student_head
        teacher_head.load_state_dict(student_head.state_dict())
        print("✓ Starting fresh training")
    
    dino_target = DINOTarget(dim=teacher_head.mlp[-1].out_features, device=device)
    if resume_checkpoint_path and Path(resume_checkpoint_path).exists():
        # Restore DINO target center if available
        if 'dino_target_center' in checkpoint:
            dino_target.center = checkpoint['dino_target_center'].to(device)
    
    num_batches = len(train_loader)
    # Use constant learning rate of 1e-4 (no scheduler)
    constant_lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = constant_lr
    
    # Initialize plot for real-time updates
    fig, ax1, ax2 = None, None, None
    plot_path = None
    if save_dir:
        plot_path = f"{save_dir}/training_curves.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        plt.ion()  # Turn on interactive mode
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        student.train()
        total_loss = 0
        
        # Use constant learning rate (no scheduler)
        lr = constant_lr
        
        # EMA momentum scheduling (DINO paper: constant 0.996 for backbone)
        # For stability, keep EMA constant or schedule very slowly
        # DINO paper uses constant 0.996, but we can schedule head EMA slightly
        # Keep backbone EMA constant at 0.996 to prevent student-teacher divergence
        current_ema_m = ema_m  # Constant 0.996 for stability

        for batch_idx, (global_crop, local_crops) in enumerate(train_loader):
            global_crop = global_crop.to(device)
            local_crops = [lc.to(device) for lc in local_crops]

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                # Student embeddings (before projection head for KoLeo)
                student_global_emb = student(global_crop, return_embedding=True)
                student_global = student_head(student_global_emb)
                # DO NOT normalize head outputs - they should be raw logits

                # Teacher embeddings (no gradient)
                with torch.no_grad():
                    teacher_global = teacher(global_crop, return_embedding=True)
                    teacher_global = teacher_head(teacher_global)
                    # DO NOT normalize head outputs - they should be raw logits
                    teacher_probs = dino_target(teacher_global)
                    
                    # Diagnostic: check teacher output variance (should be > 0)
                    if batch_idx == 0 and epoch % 20 == 0:
                        teacher_var = teacher_global.var(dim=0).mean().item()
                        teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-10)).sum(dim=1).mean().item()
                        if epoch == 0 or epoch % 50 == 0:
                            print(f"    [Debug] Teacher logits var: {teacher_var:.4f}, Teacher entropy: {teacher_entropy:.4f}")

                # DINO loss for global crop
                loss = dino_loss(student_global, teacher_probs)

                # Collect embeddings for KoLeo regularization
                embeddings_list = [student_global_emb]

                # DINO loss for local crops (student only)
                for lc in local_crops:
                    student_local_emb = student(lc, return_embedding=True)
                    student_local = student_head(student_local_emb)
                    # DO NOT normalize head outputs - they should be raw logits
                    loss += dino_loss(student_local, teacher_probs)
                    embeddings_list.append(student_local_emb)
                
                loss /= (1 + len(local_crops))
                
                # KoLeo regularization: encourage diverse representations
                if koleo_weight > 0:
                    # Concatenate all embeddings (global + local crops)
                    all_embeddings = torch.cat(embeddings_list, dim=0)  # (B*(1+num_local), embed_dim)
                    koleo_reg = koleo_loss(all_embeddings, k=3)
                    loss += koleo_weight * koleo_reg

            # Backprop
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(student_head.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            # EMA update of teacher (both backbone and head) with scheduled momentum
            update_teacher(student, teacher, student_head, teacher_head, current_ema_m)

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)  # Track loss for plotting
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {lr:.6f}, EMA: {current_ema_m:.4f}, Time: {epoch_time:.2f}s")
        
        # Update plot in real-time
        if save_dir and len(losses) > 0 and fig is not None:
            ax1.clear()
            ax1.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([1, max(len(losses), num_epochs)])
            
            if len(knn_accuracies) > 0:
                ax2.clear()
                ax2.plot(epochs_evaluated, knn_accuracies, 'r-o', linewidth=2, markersize=6)
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('k-NN Accuracy (%)', fontsize=12)
                ax2.set_title('k-NN Accuracy (Teacher)', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim([1, num_epochs])
                if len(knn_accuracies) > 0:
                    ax2.set_ylim([0, max(100, max(knn_accuracies) * 1.1)])
            else:
                ax2.clear()
                ax2.text(0.5, 0.5, 'No k-NN evaluations yet', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('k-NN Accuracy (Teacher)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.pause(0.01)  # Small pause to update plot

        # k-NN evaluation and checkpoint saving
        # DINO paper: evaluate on teacher EMA model only, using [cls] token
        knn_acc = None
        if (epoch + 1) % knn_eval_freq == 0 or (epoch + 1) == num_epochs:
            if knn_train_loader is not None and knn_test_loader is not None:
                knn_acc = knn_evaluate(teacher, knn_train_loader, knn_test_loader, k=20, device=device)
                knn_accuracies.append(knn_acc)
                epochs_evaluated.append(epoch + 1)
                print(f"--- Epoch {epoch+1}, k-NN Accuracy (teacher): {knn_acc:.2f}% ---")
                
                # Save best model based on k-NN accuracy
                if knn_acc > best_knn_acc:
                    best_knn_acc = knn_acc
                    if save_dir:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'student_state_dict': student.state_dict(),
                            'teacher_state_dict': teacher.state_dict(),
                            'student_head_state_dict': student_head.state_dict(),
                            'teacher_head_state_dict': teacher_head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'knn_acc': knn_acc,
                            'loss': avg_loss,
                            'best_knn_acc': best_knn_acc,
                            'dino_target_center': dino_target.center.cpu(),
                            'losses': losses,
                            'knn_accuracies': knn_accuracies,
                            'epochs_evaluated': epochs_evaluated,
                        }
                        torch.save(checkpoint, f"{save_dir}/best_model.pt", _use_new_zipfile_serialization=False)
                        print(f"  → Saved best model (k-NN: {knn_acc:.2f}%)")
        
        # Save latest checkpoint every epoch (for resuming)
        if save_dir:
            latest_checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'student_head_state_dict': student_head.state_dict(),
                'teacher_head_state_dict': teacher_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'knn_acc': knn_acc,
                'loss': avg_loss,
                'best_knn_acc': best_knn_acc,
                'dino_target_center': dino_target.center.cpu(),
                'losses': losses,
                'knn_accuracies': knn_accuracies,
                'epochs_evaluated': epochs_evaluated,
            }
            torch.save(latest_checkpoint, f"{save_dir}/checkpoint_latest.pt", _use_new_zipfile_serialization=False)
        
        # Periodic checkpoint saving (numbered checkpoints)
        if save_dir and (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'student_head_state_dict': student_head.state_dict(),
                'teacher_head_state_dict': teacher_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'knn_acc': knn_acc,
                'loss': avg_loss,
                'best_knn_acc': best_knn_acc,
                'dino_target_center': dino_target.center.cpu(),
                'losses': losses,
                'knn_accuracies': knn_accuracies,
                'epochs_evaluated': epochs_evaluated,
            }
            torch.save(checkpoint, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt", _use_new_zipfile_serialization=False)
            print(f"  → Saved checkpoint: checkpoint_epoch_{epoch+1}.pt")
    
    # Final plot update
    if save_dir and len(losses) > 0 and fig is not None:
        plt.ioff()  # Turn off interactive mode
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Final training curves saved to {plot_path}")
        plt.close()


# ======================================================================
# Model Architecture
# ======================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        return self.w_out(self.w1(x) * F.silu(self.w2(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(SwiGLU(embed_dim, mlp_dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )
        # Remove LayerNorm - we want raw logits, not normalized outputs

    def forward(self, x):
        x = self.mlp(x)
        return x

class VisionTransformer(nn.Module):
  def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout=0.1):
    super().__init__()
    self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
    num_patches = (image_size // patch_size) ** 2
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
    self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    self.dropout = nn.Dropout(dropout)
    self.transformer_blocks = nn.ModuleList([
      TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
    ])
    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Linear(embed_dim, num_classes)


  def forward(self, x, return_embedding: bool = False):
    batch_size = x.shape[0]
    x = self.patch_embedding(x)
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # interpolate pos embedding if sizes don't match
    if x.size(1) != self.pos_embedding.size(1):
        pos_embed = self.pos_embedding[:, 1:, :].transpose(1,2)  # (1, embed_dim, num_patches)
        H = W = int((x.size(1)-1) ** 0.5)
        pos_embed = pos_embed.reshape(1, x.size(2), int(pos_embed.size(2) ** 0.5), int(pos_embed.size(2) ** 0.5))
        pos_embed = F.interpolate(pos_embed, size=(H,W), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.flatten(2).transpose(1,2)
        pos_embed = torch.cat([self.pos_embedding[:, :1, :], pos_embed], dim=1)  # prepend cls token
    else:
        pos_embed = self.pos_embedding

    x = x + pos_embed
    x = self.dropout(x)
    for block in self.transformer_blocks:
        x = block(x)
    x = self.norm(x)
    cls_token_output = x[:, 0]  # Extract CLS token (first token) - used for k-NN evaluation
    if return_embedding:
        return cls_token_output  # Return CLS token embedding only
    logits = self.head(cls_token_output)
    return logits

# ======================================================================
# k-NN Evaluation Function (optional - only used if eval dataset provided)
# ======================================================================

@torch.no_grad()
def knn_evaluate(model, train_loader, test_loader, k, device):
    model.eval()
    # 1. Build feature bank using CLS token embeddings
    # model(images, return_embedding=True) returns the CLS token (first token) from VisionTransformer
    features_list, labels_list = [], []
    for images, labels in train_loader:
        images = images.to(device)
        feats = model(images, return_embedding=True)  # Returns CLS token embedding
        feats = F.normalize(feats, dim=1)
        features_list.append(feats.cpu())
        labels_list.append(labels)
    train_features = torch.cat(features_list, dim=0).numpy().astype('float32')
    train_labels = torch.cat(labels_list, dim=0).numpy().astype('int64')

    # 2. Build FAISS index
    d = train_features.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(train_features)

    total_correct, total_samples = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.numpy()
        feats = model(images, return_embedding=True)  # Returns CLS token embedding
        feats = F.normalize(feats, dim=1).cpu().numpy().astype('float32')
        D, I = index.search(feats, k)
        neighbor_labels = train_labels[I]
        preds = []
        for nb in neighbor_labels:
            vals, counts = np.unique(nb, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        preds = np.array(preds)
        total_correct += (preds == labels).sum()
        total_samples += labels.shape[0]

    return 100 * total_correct / total_samples


# ======================================================================
# SMALL GPU VERSION - Reduced parameters for limited GPU memory
# Configuration: embed_dim=256, num_layers=6, mlp_dim=1024, patch_size=8, batch_size=32
# ======================================================================

if __name__ == '__main__':
    ################################################################################
    ############################# DATASET SETUP ####################################
    ################################################################################

    # Load competition pretraining dataset from HuggingFace
    image_size = 96  # Competition requirement: 96x96 images
    
    # OPTIMIZATION: Use subset for faster iteration during development
    # Set to None to use full dataset, or specify number (e.g., 50000 for 50k images)
    # Full dataset: ~500k images = 2-5 hours/epoch with batch_size=32
    # 50k subset: ~20-30 min/epoch (good for testing hyperparameters)
    DATASET_SUBSET_SIZE = 50000  # None = full dataset, or set to e.g., 50000, 100000
    
    # ======================================================================
    # ENVIRONMENT DETECTION AND PATH SETUP
    # ======================================================================
    # Detect if running in Colab or on local GPU machine
    is_colab = 'google.colab' in sys.modules or os.path.exists('/content/drive')
    
    if is_colab:
        # Colab: Use Google Drive for persistence across sessions
        # Mount Drive if not already mounted
        if not os.path.exists('/content/drive/MyDrive'):
            try:
                from google.colab import drive
                drive.mount('/content/drive')
            except ImportError:
                print("⚠️  Google Drive not mounted. Please mount it manually.")
        
        DATASET_BASE_DIR = '/content/drive/MyDrive/dataset_cache'
        ZIP_DIR = os.path.join(DATASET_BASE_DIR, 'zips')
        UNZIPPED_DIR = os.path.join(DATASET_BASE_DIR, 'unzipped')
        print("🌐 Running in Google Colab - using Google Drive for dataset storage")
    else:
        # Local GPU machine: Use local directory
        DATASET_BASE_DIR = './dataset_cache'
        ZIP_DIR = os.path.join(DATASET_BASE_DIR, 'zips')
        UNZIPPED_DIR = os.path.join(DATASET_BASE_DIR, 'unzipped')
        print("💻 Running on local GPU machine - using local directory for dataset storage")
    
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(UNZIPPED_DIR, exist_ok=True)
    
    # ======================================================================
    # DOWNLOAD DATASET USING SNAPSHOT_DOWNLOAD
    # ======================================================================
    print("\n" + "="*60)
    print("Step 1: Downloading dataset ZIP files...")
    print("="*60)
    
    dataset_repo = "tsbpp/fall2025_deeplearning"
    
    # Check if ZIP files already exist
    existing_zips = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
    if existing_zips:
        print(f"✓ Found {len(existing_zips)} existing ZIP files in {ZIP_DIR}")
        print("  Skipping download (files already exist)")
    else:
        print(f"Downloading dataset from {dataset_repo} to {ZIP_DIR}...")
        print("  This may take 10-20 minutes depending on your connection...")
        try:
            snapshot_download(
                repo_id=dataset_repo,
                local_dir=ZIP_DIR,
                repo_type="dataset",
                ignore_patterns=["*.md", "*.txt"]  # Skip README files
            )
            print("✓ Download complete!")
        except Exception as e:
            print(f"⚠️  Error during download: {e}")
            print("  Will try to use existing files or fall back to load_dataset()...")
    
    # ======================================================================
    # UNZIP FILES
    # ======================================================================
    print("\n" + "="*60)
    print("Step 2: Unzipping files...")
    print("="*60)
    
    # Find all ZIP files
    zip_files = glob.glob(os.path.join(ZIP_DIR, "*.zip"))
    
    # Track if we're using unzipped images or fallback to load_dataset
    use_unzipped = False
    
    if not zip_files:
        print("⚠️  No ZIP files found! Falling back to load_dataset()...")
        use_unzipped = False
    else:
        use_unzipped = True
        # Check if already unzipped (look for image files in unzipped directory)
        existing_images = glob.glob(os.path.join(UNZIPPED_DIR, "**", "*.jpg"), recursive=True)
        existing_images += glob.glob(os.path.join(UNZIPPED_DIR, "**", "*.png"), recursive=True)
        existing_images += glob.glob(os.path.join(UNZIPPED_DIR, "**", "*.jpeg"), recursive=True)
        
        if existing_images:
            print(f"✓ Found {len(existing_images):,} existing unzipped images in {UNZIPPED_DIR}")
            print("  Skipping unzip (files already extracted)")
        else:
            print(f"Unzipping {len(zip_files)} ZIP files to {UNZIPPED_DIR}...")
            print("  This will take approximately 1 hour, but only needs to be done once!")
            
            for i, zip_path in enumerate(zip_files, 1):
                zip_name = os.path.basename(zip_path)
                print(f"  [{i}/{len(zip_files)}] Unzipping {zip_name}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(UNZIPPED_DIR)
                    print(f"    ✓ {zip_name} extracted")
                except Exception as e:
                    print(f"    ⚠️  Error unzipping {zip_name}: {e}")
            
            print("✓ Unzip complete!")
        
        # ======================================================================
        # CREATE CUSTOM DATASET LOADER FOR UNZIPPED IMAGES
        # ======================================================================
        print("\n" + "="*60)
        print("Step 3: Loading images from unzipped directory...")
        print("="*60)
        
        class ImageFolderDataset(torch.utils.data.Dataset):
            """Custom dataset that loads images from a directory of unzipped images."""
            def __init__(self, image_dir, image_size=96):
                """
                Args:
                    image_dir: Directory containing unzipped images
                    image_size: Target image size (for validation)
                """
                self.image_dir = Path(image_dir)
                self.image_size = image_size
                
                # Find all image files recursively
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                self.image_paths = []
                for ext in image_extensions:
                    self.image_paths.extend(self.image_dir.rglob(ext))
                
                # Sort for reproducibility
                self.image_paths = sorted(self.image_paths)
                
                print(f"  Found {len(self.image_paths):,} images in {image_dir}")
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                """Return PIL Image (compatible with existing DINODataset)."""
                img_path = self.image_paths[idx]
                try:
                    img = Image.open(img_path).convert('RGB')
                    return {'image': img}  # Return dict format like HuggingFace dataset
                except Exception as e:
                    print(f"⚠️  Error loading {img_path}: {e}")
                    # Return a black image as fallback
                    return {'image': Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))}
        
        # Create dataset from unzipped images
        pretrain_dataset = ImageFolderDataset(UNZIPPED_DIR, image_size=image_size)
        
        # Optionally use a subset for faster iteration
        if DATASET_SUBSET_SIZE is not None and len(pretrain_dataset) > DATASET_SUBSET_SIZE:
            print(f"⚠️  Using SUBSET of {DATASET_SUBSET_SIZE:,} images for faster iteration")
            # Create a subset by selecting first N images
            class SubsetDataset(torch.utils.data.Dataset):
                def __init__(self, dataset, indices):
                    self.dataset = dataset
                    self.indices = indices
                def __len__(self):
                    return len(self.indices)
                def __getitem__(self, idx):
                    return self.dataset[self.indices[idx]]
            
            subset_indices = list(range(min(DATASET_SUBSET_SIZE, len(pretrain_dataset))))
            pretrain_dataset = SubsetDataset(pretrain_dataset, subset_indices)
        
        print(f"✓ Loaded {len(pretrain_dataset):,} images from unzipped directory")
        print(f"  Dataset location: {UNZIPPED_DIR}")
    
    # Fallback to load_dataset if unzipping failed or no ZIP files
    if not use_unzipped:
        print("\n" + "="*60)
        print("Using load_dataset() fallback method...")
        print("="*60)
        pretrain_dataset = load_dataset(dataset_repo, split="train")
        if DATASET_SUBSET_SIZE is not None:
            print(f"⚠️  Using SUBSET of {DATASET_SUBSET_SIZE:,} images for faster iteration")
            pretrain_dataset = pretrain_dataset.select(range(min(DATASET_SUBSET_SIZE, len(pretrain_dataset))))
        print(f"Loaded {len(pretrain_dataset):,} images using load_dataset()")

    # ----------------------------
    # DINO-style SSL dataset
    # ----------------------------
    # OPTIMIZATION: Reduce num_local_crops to speed up training
    # - 4 local crops: Standard DINO (slower, better quality)
    # - 2 local crops: ~40% faster, slightly less effective
    # - 1 local crop: ~60% faster, less effective
    num_local_crops = 4  # Standard DINO (set to 2 for faster training)
    
    class DINODataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, image_size=96, num_local_crops=4):
            """
            Args:
                hf_dataset: HuggingFace dataset (returns dict with 'image' key)
                image_size: Target image size
                num_local_crops: Number of local crops per image
            """
            self.dataset = hf_dataset
            self.num_local_crops = num_local_crops
            
            # Global crop transforms (DINO-style augmentations)
            global_transforms = [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomApply([transforms.RandomSolarize(threshold=128, p=1.0)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.global_transform = transforms.Compose(global_transforms)
            
            # Local crop transforms
            local_transforms = [
                transforms.RandomResizedCrop(image_size, scale=(0.14, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomApply([transforms.RandomSolarize(threshold=128, p=1.0)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.local_transform = transforms.Compose(local_transforms)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # HuggingFace dataset returns dict with 'image' key (PIL Image)
            item = self.dataset[idx]
            img = item['image']
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to ensure minimum size for cropping
            if min(img.size) < image_size:
                img = transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC)(img)
            
            # Apply transforms
            global_crop = self.global_transform(img)
            local_crops = [self.local_transform(img) for _ in range(self.num_local_crops)]
            
            return global_crop, local_crops

    # Create SSL dataset from HuggingFace pretraining data
    ssl_ds_train = DINODataset(pretrain_dataset, image_size=image_size, num_local_crops=num_local_crops)
    print(f"Using {num_local_crops} local crops per image (1 global + {num_local_crops} local = {num_local_crops+1} total crops)")

    # ----------------------------
    # DataLoaders
    # ----------------------------
    # IMPORTANT: Use 8-12 workers for parallel I/O and augmentation
    # This is critical for HuggingFace datasets which load from ZIP files
    # More workers = better parallelization of image loading and transforms
    num_workers = 12  # Recommended: 8-12 for Colab/GPU machines with good CPUs
    # Alternative: num_workers = min(12, os.cpu_count() or 2)  # Use up to 12 or available cores
    print(f"Using {num_workers} workers for DataLoaders (critical for fast I/O from ZIP files).")

    # SMALL GPU VERSION: Reduced batch size for limited GPU memory
    # DINO authors: LR is most sensitive hyperparameter, scale with batch size
    # Smaller batch sizes need proportionally smaller learning rates
    batch_size = 32  # SMALL GPU VERSION: Reduced for limited GPU memory
    # Try increasing to 64 or 128 if you have more GPU memory (faster training)
    
    # Calculate estimated epoch time
    iterations_per_epoch = len(ssl_ds_train) / batch_size
    estimated_hours = iterations_per_epoch * 0.8 / 3600  # Rough estimate: 0.8s per iteration
    print(f"\n📊 Training info:")
    print(f"   Batch size: {batch_size}")
    print(f"   Iterations per epoch: ~{iterations_per_epoch:,.0f}")
    print(f"   Estimated time per epoch: ~{estimated_hours:.1f} hours (varies by GPU)")
    print(f"   Tip: Reduce DATASET_SUBSET_SIZE or num_local_crops to speed up")
    
    train_loader = DataLoader(ssl_ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    dataset_name = "Competition Pretraining Dataset (HuggingFace)"
    print(f"{dataset_name} loaded. SSL Train: {len(ssl_ds_train)} images")

    ################################################################################
    ############################# MODEL SETUP ######################################
    ################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- Vision Transformer student + teacher ---
    # SMALL GPU VERSION: Reduced parameters for limited GPU memory
    # With 96x96 images: patch_size=8 gives 144 patches (vs 576 with patch_size=4)
    embed_dim = 256  # SMALL GPU VERSION: Reduced from 512
    model = VisionTransformer(
        image_size=image_size,  # 96x96 for competition
        patch_size=8,  # SMALL GPU VERSION: Larger patches = fewer: (96/8)^2 = 144 patches
        in_channels=3,
        embed_dim=embed_dim,
        num_heads=8,    # embed_dim must be divisible by num_heads (256/8 = 32)
        mlp_dim=1024,   # SMALL GPU VERSION: Reduced (4x embed_dim)
        num_layers=6,  # SMALL GPU VERSION: Reduced from 12
        num_classes=100,  # not used for SSL
        dropout=0.1
    ).to(device)

    student = model
    teacher = copy.deepcopy(student)
    teacher.eval()

    # --- Projection heads ---
    # DINO paper: use >=65k prototypes for best results
    # Using 65536 (2^16) for optimal performance - this is critical for stability
    projection_dim = 65536
    student_head = DINOHead(in_dim=embed_dim, out_dim=projection_dim).to(device)
    teacher_head = DINOHead(in_dim=embed_dim, out_dim=projection_dim).to(device)

    # --- Optimizer ---
    # Use constant learning rate of 1e-4 (no scheduler, no batch size scaling)
    learning_rate = 1e-4
    
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(student_head.parameters()),
        lr=learning_rate,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )
    print(f"Using constant learning rate: {learning_rate:.6f}")

    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    ################################################################################
    ############################# TRAINING ########################################
    ################################################################################

    print("Start Time:" + time.strftime("%H:%M:%S", time.localtime()))
    print("Starting DINO training...")

    # DINO paper trains for 300 epochs on ImageNet
    # For competition dataset, we'll use 200 epochs
    # Optional: Resume from a specific checkpoint
    # Set resume_from=None to start fresh, or provide path like "./checkpoints/checkpoint_latest.pt"
    resume_from = None  # Change to checkpoint path if you want to resume from a specific checkpoint
    
    train_dino(
        train_loader,
        student,
        teacher,
        student_head,
        teacher_head,
        optimizer,
        device=device,
        num_epochs=200,  # DINO paper uses 300, but smaller datasets need fewer epochs
        ema_m=0.996,
        knn_eval_freq=20,  # k-NN evaluation disabled (no eval dataset)
        warmup_epochs=10,  # Not used anymore (constant LR), but kept for compatibility
        save_dir="./checkpoints",  # Save checkpoints here
        save_freq=10,  # Save checkpoint every 10 epochs
        koleo_weight=0.1,  # DINOv2: KoLeo regularization weight (0.1 is a good default)
        knn_train_loader=None,  # No k-NN evaluation dataset
        knn_test_loader=None,  # No k-NN evaluation dataset
        resume_from=resume_from  # Resume from checkpoint if interrupted
    )

    print("End Time:" + time.strftime("%H:%M:%S", time.localtime()))

