#!/usr/bin/env python3
"""
Quick script to check actual image sizes in the HuggingFace dataset
and compare loading performance with CIFAR-10
"""

import time
from datasets import load_dataset
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

print("=" * 60)
print("Checking HuggingFace Dataset Image Sizes")
print("=" * 60)

# Load HuggingFace dataset
print("\n1. Loading HuggingFace dataset...")
start = time.time()
pretrain_dataset = load_dataset("tsbpp/fall2025_deeplearning", split="train")
print(f"   Loaded in {time.time() - start:.2f}s")
print(f"   Total images: {len(pretrain_dataset):,}")

# Check first 100 images
print("\n2. Checking image sizes (first 100 images)...")
sizes = []
load_times = []

for i in range(min(100, len(pretrain_dataset))):
    load_start = time.time()
    item = pretrain_dataset[i]
    img = item['image']
    load_time = time.time() - load_start
    load_times.append(load_time)
    
    if isinstance(img, Image.Image):
        sizes.append(img.size)
    else:
        sizes.append("Unknown type")
    
    if i < 5:
        print(f"   Image {i}: {img.size if isinstance(img, Image.Image) else type(img)}, "
              f"mode={img.mode if isinstance(img, Image.Image) else 'N/A'}, "
              f"load_time={load_time*1000:.2f}ms")

# Statistics
if sizes and all(isinstance(s, tuple) for s in sizes):
    sizes_array = np.array(sizes)
    print(f"\n3. Image Size Statistics (first 100):")
    print(f"   Width:  min={sizes_array[:, 0].min()}, max={sizes_array[:, 0].max()}, "
          f"mean={sizes_array[:, 0].mean():.1f}, median={np.median(sizes_array[:, 0]):.1f}")
    print(f"   Height: min={sizes_array[:, 1].min()}, max={sizes_array[:, 1].max()}, "
          f"mean={sizes_array[:, 1].mean():.1f}, median={np.median(sizes_array[:, 1]):.1f}")
    print(f"   Unique sizes: {len(set(sizes))}")
    print(f"   Most common size: {max(set(sizes), key=sizes.count)}")

print(f"\n4. Loading Performance (first 100):")
print(f"   Average load time: {np.mean(load_times)*1000:.2f}ms")
print(f"   Median load time: {np.median(load_times)*1000:.2f}ms")
print(f"   Max load time: {np.max(load_times)*1000:.2f}ms")
print(f"   Min load time: {np.min(load_times)*1000:.2f}ms")

print("\n" + "=" * 60)
print("Comparing with CIFAR-10 Loading")
print("=" * 60)

# Load CIFAR-10 for comparison
print("\n5. Loading CIFAR-10 dataset...")
start = time.time()
cifar10_train = CIFAR10(root='/tmp/cifar10', train=True, download=True, transform=None)
print(f"   Loaded in {time.time() - start:.2f}s")
print(f"   Total images: {len(cifar10_train):,}")

# Check CIFAR-10 loading times
print("\n6. Checking CIFAR-10 image sizes and load times (first 100)...")
cifar_sizes = []
cifar_load_times = []

for i in range(min(100, len(cifar10_train))):
    load_start = time.time()
    img, _ = cifar10_train[i]
    load_time = time.time() - load_start
    cifar_load_times.append(load_time)
    
    if isinstance(img, Image.Image):
        cifar_sizes.append(img.size)
    else:
        cifar_sizes.append("Unknown type")
    
    if i < 5:
        print(f"   Image {i}: {img.size if isinstance(img, Image.Image) else type(img)}, "
              f"load_time={load_time*1000:.2f}ms")

if cifar_sizes and all(isinstance(s, tuple) for s in cifar_sizes):
    cifar_sizes_array = np.array(cifar_sizes)
    print(f"\n7. CIFAR-10 Size Statistics (first 100):")
    print(f"   Width:  min={cifar_sizes_array[:, 0].min()}, max={cifar_sizes_array[:, 0].max()}, "
          f"mean={cifar_sizes_array[:, 0].mean():.1f}")
    print(f"   Height: min={cifar_sizes_array[:, 1].min()}, max={cifar_sizes_array[:, 1].max()}, "
          f"mean={cifar_sizes_array[:, 1].mean():.1f}")

print(f"\n8. CIFAR-10 Loading Performance (first 100):")
print(f"   Average load time: {np.mean(cifar_load_times)*1000:.2f}ms")
print(f"   Median load time: {np.median(cifar_load_times)*1000:.2f}ms")
print(f"   Max load time: {np.max(cifar_load_times)*1000:.2f}ms")
print(f"   Min load time: {np.min(cifar_load_times)*1000:.2f}ms")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
if load_times and cifar_load_times:
    speedup = np.mean(cifar_load_times) / np.mean(load_times)
    print(f"HuggingFace images load {speedup:.2f}x {'slower' if speedup < 1 else 'faster'} than CIFAR-10")
    print(f"  (HuggingFace: {np.mean(load_times)*1000:.2f}ms vs CIFAR-10: {np.mean(cifar_load_times)*1000:.2f}ms)")

