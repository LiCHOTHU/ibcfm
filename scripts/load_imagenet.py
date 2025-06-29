import os
import pickle
import argparse
import numpy as np
import imageio
from tqdm import tqdm

def load_batch(path):
    """Unpickle a batch file and return its contents."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Convert ImageNet-64 pickled batches to ImageFolder PNGs")
    parser.add_argument('--data_dir', required=True,
                        help="Path containing train_data_batch_1…10 and val_data files")
    parser.add_argument('--out_dir', default='imagenet64_png',
                        help="Output root for train/val PNG folders")
    parser.add_argument('--img_size', type=int, default=64,
                        help="Height/width of images (default: 64)")
    parser.add_argument('--num_classes', type=int, default=1000,
                        help="Number of classes (default: 1000)")
    args = parser.parse_args()

    # Ensure input exists
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Create output root
    os.makedirs(args.out_dir, exist_ok=True)

    # Process validation split
    print("Converting validation split...")
    val_path = os.path.join(args.data_dir, 'val_data')
    val_batch = load_batch(val_path)
    val_imgs  = np.array(val_batch['data'], dtype=np.uint8) \
                   .reshape(-1, 3, args.img_size, args.img_size) \
                   .transpose(0, 2, 3, 1)
    val_labels = val_batch['labels']
    for i, img in enumerate(tqdm(val_imgs, desc="  val")):
        lbl = val_labels[i] - 1 if min(val_labels) >= 1 else val_labels[i]
        out_dir = os.path.join(args.out_dir, 'val', f"{lbl:05d}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{i:08d}.png")
        imageio.imwrite(out_path, img)

    # Process training splits
    print("Converting training split...")
    counter = 0
    for b in range(1, 11):
        batch_file = os.path.join(args.data_dir, f"train_data_batch_{b}")
        batch = load_batch(batch_file)
        imgs = np.array(batch['data'], dtype=np.uint8) \
                  .reshape(-1, 3, args.img_size, args.img_size) \
                  .transpose(0, 2, 3, 1)
        labels = batch['labels']
        for img, label in tqdm(zip(imgs, labels), desc=f"  train batch {b}", total=len(labels)):
            lbl = label - 1 if min(labels) >= 1 else label
            out_dir = os.path.join(args.out_dir, 'train', f"{lbl:05d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{counter:08d}.png")
            imageio.imwrite(out_path, img)
            counter += 1

    print(f"Conversion complete. PNGs saved to: {args.out_dir}")

if __name__ == "__main__":
    main()

