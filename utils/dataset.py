import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_imagenet64_loaders(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4
):
    """
    Create DataLoaders for 64×64 ImageNet saved in ImageFolder format.

    Args:
        data_root: Path to root folder containing 'train' and 'val' subdirs.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to pin memory in DataLoader for faster GPU transfer.
        prefetch_factor: Number of batches to prefetch per worker (default 4).

    Returns:
        train_loader, val_loader: PyTorch DataLoader objects.
    """
    train_dir = os.path.join(data_root, 'train')
    val_dir   = os.path.join(data_root, 'val')

    # Training transforms: random crop + flip + normalize
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # <— change these two lines only:
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # <— and change these too:
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    full_val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    # Subsample validation
    val_fraction = 0.1
    val_size = len(full_val_ds)
    small_val_sz = max(1, int(val_size * val_fraction))
    val_ds = Subset(full_val_ds, list(range(small_val_sz)))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    return train_loader, val_loader

# Example usage:
if __name__ == "__main__":
    data_root = "/mnt/sda3/Downloads/imagenet64_png"
    train_loader, val_loader = get_imagenet64_loaders(
        data_root,
        batch_size=128,
        num_workers=16
    )
    print(f"Loaded {len(train_loader.dataset)} training and {len(val_loader.dataset)} validation samples")

