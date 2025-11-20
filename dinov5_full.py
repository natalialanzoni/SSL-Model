# DINO Self-Supervised Learning Implementation
# Converted from Jupyter notebook
#
# USAGE:
#   python dinov5_full.py
#
# This script will automatically check and install required dependencies.
# For manual installation, use: pip install -r requirements.txt
#
# REQUIREMENTS:
#   - Python 3.8+
#   - CUDA-capable GPU recommended (A100 40GB, V100 32GB, or similar)
#   - ~16GB+ GPU memory for full model
#
# The script will:
#   1. Check and install dependencies automatically
#   2. Load and combine CIFAR-10 and STL-10 datasets
#   3. Train DINO model with constant learning rate (1e-4)
#   4. Generate training curves (loss and k-NN accuracy plots)
#   5. Save checkpoints and best model

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
import numpy as np
import os # --- NEW --- Import os to get CPU count
import matplotlib.pyplot as plt  # For plotting loss and k-NN accuracy

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
               knn_train_loader=None, knn_test_loader=None):

    # Initialize teacher as a copy of student
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    # Initialize teacher_head as a copy of student_head
    teacher_head.load_state_dict(student_head.state_dict())
    dino_target = DINOTarget(dim=teacher_head.mlp[-1].out_features, device=device)
    
    # Create checkpoint directory
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    best_knn_acc = 0.0
    
    # Track losses and k-NN accuracies for plotting
    losses = []
    knn_accuracies = []
    epochs_evaluated = []

    scaler = torch.amp.GradScaler()
    num_batches = len(train_loader)
    # Use constant learning rate of 1e-4 (no scheduler)
    constant_lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = constant_lr
    
    for epoch in range(num_epochs):
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

        # k-NN evaluation and checkpoint saving
        # DINO paper: evaluate on teacher EMA model only, using [cls] token
        knn_acc = None
        if (epoch + 1) % knn_eval_freq == 0 or (epoch + 1) == num_epochs:
            if knn_train_loader is not None and knn_test_loader is not None:
                knn_acc = knn_evaluate(teacher, knn_train_loader, knn_test_loader, k=20, device=device)
                knn_accuracies.append(knn_acc)
                epochs_evaluated.append(epoch + 1)
                print(f"--- Epoch {epoch+1}, k-NN Accuracy (teacher): {knn_acc:.2f}% ---")
            
            # Save best model
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
                        'knn_acc': knn_acc,
                        'loss': avg_loss,
                        'dino_target_center': dino_target.center.cpu(),
                    }
                    torch.save(checkpoint, f"{save_dir}/best_model.pt")
                    print(f"  → Saved best model (k-NN: {knn_acc:.2f}%)")
        
        # Periodic checkpoint saving
        if save_dir and (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': student.state_dict(),
                'teacher_state_dict': teacher.state_dict(),
                'student_head_state_dict': student_head.state_dict(),
                'teacher_head_state_dict': teacher_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'knn_acc': knn_acc,
                'loss': avg_loss,
                'dino_target_center': dino_target.center.cpu(),
            }
            torch.save(checkpoint, f"{save_dir}/checkpoint_epoch_{epoch+1}.pt")
    
    # Plot loss and k-NN accuracy
    if len(losses) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([1, len(losses)])
        
        # Plot k-NN accuracy
        if len(knn_accuracies) > 0:
            ax2.plot(epochs_evaluated, knn_accuracies, 'r-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('k-NN Accuracy (%)', fontsize=12)
            ax2.set_title('k-NN Accuracy (Teacher)', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim([1, num_epochs])
            ax2.set_ylim([0, max(100, max(knn_accuracies) * 1.1)])
        else:
            ax2.text(0.5, 0.5, 'No k-NN evaluations yet', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('k-NN Accuracy (Teacher)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_path = f"{save_dir}/training_curves.png" if save_dir else "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved training curves to {plot_path}")
        plt.close()


# ======================================================================
# Cell 1
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

# ======================================================================
# Cell 2
# ======================================================================

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

# ======================================================================
# Cell 3
# ======================================================================

class DINOSystem(nn.Module):
    def __init__(self, vit_student, vit_teacher, embed_dim, out_dim=65536):
        super().__init__()

        self.student = vit_student
        self.teacher = vit_teacher

        # teacher not trained directly
        for p in self.teacher.parameters():
            p.requires_grad = False

        # projection heads
        self.student_head = DINOHead(embed_dim, out_dim)
        self.teacher_head = DINOHead(embed_dim, out_dim)

# ======================================================================
# Cell 5
# ======================================================================

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
# Cell 6
# ======================================================================

# --- NEW --- k-NN Evaluation Function
# DINO paper: evaluate on teacher EMA model, using [cls] token only
# The return_embedding=True flag returns cls_token_output = x[:, 0] from VisionTransformer
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
# Cell 7
# ======================================================================

# ======================================================================
# FULL SIZE VERSION - Original parameters for larger GPUs
# Configuration: embed_dim=512, num_layers=12, mlp_dim=2048, patch_size=4, batch_size=128
# Recommended for: A100 40GB, V100 32GB, or similar high-memory GPUs
# ======================================================================

if __name__ == '__main__':
    ################################################################################
    ############################# DEPENDENCY INSTALLATION #########################
    ################################################################################
    
    import subprocess
    import sys
    
    def install_package(package):
        """Install a package using pip if not already installed"""
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
    
    print("=" * 70)
    print("CHECKING AND INSTALLING DEPENDENCIES")
    print("=" * 70)
    
    # Core dependencies
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'faiss': 'faiss-cpu',  # Use faiss-gpu if you have CUDA and want GPU acceleration
        'datasets': 'datasets',  # HuggingFace datasets (for load_dataset if needed)
    }
    
    # Try to import, install if missing
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'PIL':
                import PIL
            elif module_name == 'faiss':
                import faiss
            elif module_name == 'datasets':
                from datasets import load_dataset  # Test HuggingFace datasets import
            else:
                __import__(module_name)
            print(f"✓ {package_name} already installed")
        except ImportError:
            print(f"Installing {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✓ {package_name} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package_name}. Please install manually: pip install {package_name}")
                sys.exit(1)
    
    # Optional: Check for GPU and suggest faiss-gpu if available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            try:
                import faiss
                # Check if faiss has GPU support
                if not hasattr(faiss, 'StandardGpuResources'):
                    print("\n  Note: For better k-NN performance, consider installing faiss-gpu:")
                    print("    pip uninstall faiss-cpu")
                    print("    pip install faiss-gpu")
            except:
                pass
        else:
            print("\n⚠ CUDA not available. Training will use CPU (much slower).")
            print("  For GPU training, ensure CUDA and PyTorch with CUDA support are installed.")
    except:
        pass
    
    print("=" * 70)
    print("ALL DEPENDENCIES CHECKED - READY TO RUN")
    print("=" * 70)
    print()
    
    # Additional imports needed for main execution
    from torchvision.transforms import InterpolationMode
    from torchvision.datasets import STL10, CIFAR10
    import copy

    ################################################################################
    ############################# DATASET SETUP ####################################
    ################################################################################

    # Use both CIFAR-10 and STL-10 datasets combined
    image_size = 96  # Competition requirement: 96x96 images
    
    print("Loading CIFAR-10 dataset (will resize 32x32 -> 96x96)...")
    cifar10_root = '/tmp/cifar10'
    cifar10_train = CIFAR10(root=cifar10_root, train=True, download=True, transform=None)
    cifar10_test = CIFAR10(root=cifar10_root, train=False, download=True, transform=None)
    
    print("Loading STL-10 dataset (96x96 native)...")
    stl10_root = '/tmp/stl10'
    stl10_train = STL10(root=stl10_root, split='train', download=True, transform=None)
    stl10_test = STL10(root=stl10_root, split='test', download=True, transform=None)
    
    # Combine datasets using ConcatDataset
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([cifar10_train, stl10_train])
    test_ds = ConcatDataset([cifar10_test, stl10_test])
    
    print(f"Combined dataset: CIFAR-10 ({len(cifar10_train)} train, {len(cifar10_test)} test) + "
          f"STL-10 ({len(stl10_train)} train, {len(stl10_test)} test)")
    print(f"Total: {len(train_ds)} train, {len(test_ds)} test samples")
    
    # CIFAR-10 needs resizing, STL-10 doesn't - we'll handle this in DINODataset
    resize_input = True  # Will resize CIFAR-10, but STL-10 is already 96x96

    # ----------------------------
    # DINO-style SSL dataset
    # ----------------------------
    class DINODataset(torch.utils.data.Dataset):
        def __init__(self, torchvision_dataset, image_size=96, num_local_crops=4, 
                     cifar10_size=None, stl10_size=None):
            """
            Args:
                torchvision_dataset: Can be a single dataset or ConcatDataset
                cifar10_size: Size of CIFAR-10 dataset (if combined with STL-10)
                stl10_size: Size of STL-10 dataset (if combined with CIFAR-10)
            """
            self.dataset = torchvision_dataset
            self.num_local_crops = num_local_crops
            self.cifar10_size = cifar10_size  # CIFAR-10 needs resizing
            self.stl10_size = stl10_size  # STL-10 is already 96x96
            
            # Transforms for CIFAR-10 (needs resizing from 32x32 to 96x96)
            global_transforms_cifar = [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomApply([transforms.RandomSolarize(threshold=128, p=1.0)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.global_transform_cifar = transforms.Compose(global_transforms_cifar)
            
            local_transforms_cifar = [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomResizedCrop(image_size, scale=(0.14, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomApply([transforms.RandomSolarize(threshold=128, p=1.0)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.local_transform_cifar = transforms.Compose(local_transforms_cifar)
            
            # Transforms for STL-10 (already 96x96, no resizing needed)
            global_transforms_stl = [
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomApply([transforms.RandomSolarize(threshold=128, p=1.0)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.global_transform_stl = transforms.Compose(global_transforms_stl)
            
            local_transforms_stl = [
                transforms.RandomResizedCrop(image_size, scale=(0.14, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomApply([transforms.RandomSolarize(threshold=128, p=1.0)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            self.local_transform_stl = transforms.Compose(local_transforms_stl)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # torchvision datasets return (image, label) tuples
            img, _ = self.dataset[idx]  # We don't need the label for SSL
            
            # Determine if this is from CIFAR-10 (needs resize) or STL-10 (already 96x96)
            if self.cifar10_size is not None and idx < self.cifar10_size:
                # CIFAR-10 sample - use transforms with resizing
                global_crop = self.global_transform_cifar(img)
                local_crops = [self.local_transform_cifar(img) for _ in range(self.num_local_crops)]
            else:
                # STL-10 sample - use transforms without resizing
                global_crop = self.global_transform_stl(img)
                local_crops = [self.local_transform_stl(img) for _ in range(self.num_local_crops)]
            
            return global_crop, local_crops

    ssl_ds_train = DINODataset(train_ds, image_size=image_size, num_local_crops=4,
                               cifar10_size=len(cifar10_train), stl10_size=len(stl10_train))

    # ----------------------------
    # Evaluation datasets (k-NN)
    # ----------------------------
    # CIFAR-10 needs resizing, STL-10 doesn't
    eval_transform_cifar = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform_stl = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # For evaluation, we need to recreate the datasets with the eval transform
    # Combine both CIFAR-10 and STL-10 for k-NN evaluation
    knn_cifar10_train = CIFAR10(root=cifar10_root, train=True, download=False, transform=eval_transform_cifar)
    knn_cifar10_test = CIFAR10(root=cifar10_root, train=False, download=False, transform=eval_transform_cifar)
    knn_stl10_train = STL10(root=stl10_root, split='train', download=False, transform=eval_transform_stl)
    knn_stl10_test = STL10(root=stl10_root, split='test', download=False, transform=eval_transform_stl)
    
    knn_train_ds = ConcatDataset([knn_cifar10_train, knn_stl10_train])
    knn_test_ds = ConcatDataset([knn_cifar10_test, knn_stl10_test])

    # ----------------------------
    # DataLoaders
    # ----------------------------
    num_cores = os.cpu_count() or 2
    print(f"Using {num_cores} workers for DataLoaders.")

    # FULL SIZE VERSION: Batch size optimized for larger GPUs
    # DINO authors: LR is most sensitive hyperparameter, scale with batch size
    batch_size = 128  # FULL SIZE VERSION: Optimal for larger GPUs
    train_loader = DataLoader(ssl_ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_cores, pin_memory=True)
    knn_train_loader = DataLoader(knn_train_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_cores, pin_memory=True)
    knn_test_loader = DataLoader(knn_test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_cores, pin_memory=True)

    dataset_name = "CIFAR-10 + STL-10 (Combined)"
    print(f"{dataset_name} loaded. SSL Train: {len(ssl_ds_train)}, k-NN Train: {len(knn_train_ds)}, k-NN Test: {len(knn_test_ds)}")

    ################################################################################
    ############################# MODEL SETUP ######################################
    ################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- Vision Transformer student + teacher ---
    # FULL SIZE VERSION: Optimized for larger GPUs (A100 40GB, V100 32GB+)
    # With 96x96 images: patch_size=4 gives 576 patches, patch_size=8 gives 144 patches
    # Using patch_size=4 for better detail capture (more patches = finer granularity)
    embed_dim = 512  # FULL SIZE VERSION: Full capacity for best performance
    model = VisionTransformer(
        image_size=image_size,  # 96x96 for competition
        patch_size=4,  # FULL SIZE VERSION: Smaller patches = more detail: (96/4)^2 = 576 patches
        in_channels=3,
        embed_dim=embed_dim,
        num_heads=8,    # embed_dim must be divisible by num_heads (512/8 = 64)
        mlp_dim=2048,   # FULL SIZE VERSION: Full MLP capacity (4x embed_dim)
        num_layers=12,  # FULL SIZE VERSION: Full depth for best performance
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
    # For CIFAR-10, we'll use 200 epochs (smaller dataset needs fewer epochs)
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
        knn_eval_freq=20,
        warmup_epochs=10,  # Not used anymore (constant LR), but kept for compatibility
        save_dir="./checkpoints",  # Save checkpoints here
        save_freq=10,  # Save checkpoint every 10 epochs
        koleo_weight=0.1,  # DINOv2: KoLeo regularization weight (0.1 is a good default)
        knn_train_loader=knn_train_loader,  # Pass k-NN loaders for evaluation
        knn_test_loader=knn_test_loader
    )

    print("End Time:" + time.strftime("%H:%M:%S", time.localtime()))
