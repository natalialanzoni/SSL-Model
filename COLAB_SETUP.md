# Google Colab Setup Guide for DINO Training

## Problem
HuggingFace datasets load from ZIP files which is **very slow** (~1.4 seconds per image). This makes training extremely slow.

## Solution: Use Google Drive for Caching

### Option 1: Download to Google Drive (Recommended)

**Step 1: Mount Google Drive in Colab**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 2: Set up cache directory on Drive**
```python
import os
# Create cache directory on Drive
CACHE_DIR = '/content/drive/MyDrive/huggingface_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
```

**Step 3: Load dataset with cache_dir parameter**
```python
from datasets import load_dataset

# This will download to Drive and cache there
pretrain_dataset = load_dataset(
    "tsbpp/fall2025_deeplearning", 
    split="train",
    cache_dir=CACHE_DIR  # Cache to Google Drive
)
```

**Benefits:**
- Dataset downloads once to Drive (persists across Colab sessions)
- No need to re-download each time
- Still loads from ZIP (slow), but at least it's cached

### Option 2: Download Locally, Upload to Drive (If you have fast local internet)

**Step 1: Download dataset on your computer**
```bash
# On your local machine
python -c "from datasets import load_dataset; load_dataset('tsbpp/fall2025_deeplearning', split='train', cache_dir='./hf_cache')"
```

**Step 2: Upload to Google Drive**
- Upload the entire `hf_cache` folder to Google Drive
- Or upload just the dataset folder: `hf_cache/datasets/tsbpp___fall2025_deeplearning/...`

**Step 3: In Colab, mount Drive and use cache_dir**
```python
from google.colab import drive
drive.mount('/content/drive')

from datasets import load_dataset
pretrain_dataset = load_dataset(
    "tsbpp/fall2025_deeplearning",
    split="train",
    cache_dir='/content/drive/MyDrive/hf_cache'  # Point to uploaded cache
)
```

## Important Notes

1. **ZIP files are still slow**: Even with caching, loading from ZIP files is slow (~1.4s per image). This is why you need **12 workers** in DataLoader to parallelize.

2. **First download takes time**: The dataset is ~2.6GB, so first download will take 5-10 minutes depending on your connection.

3. **Drive storage**: Make sure you have enough space on Google Drive (dataset is ~2.6GB compressed, ~3-4GB when extracted/cached).

4. **Workers are critical**: With 12 workers, you can load 12 images in parallel, which helps offset the slow ZIP loading.

## Quick Start Code for Colab

Add this at the top of your notebook:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set up cache directory
import os
CACHE_DIR = '/content/drive/MyDrive/huggingface_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variable (optional, for all HuggingFace operations)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
```

Then modify your `load_dataset` call:
```python
pretrain_dataset = load_dataset(
    "tsbpp/fall2025_deeplearning",
    split="train",
    cache_dir=CACHE_DIR
)
```

