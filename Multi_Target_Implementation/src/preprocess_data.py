# -*- coding: utf-8 -*-
#
# preprocess_data.py
# ------------------------------------------------------
# Script for offline data augmentation of sequential recommendation datasets.
# Generates augmented sequence files to be used during training.
# ------------------------------------------------------

import os
import argparse
import shutil
from data_augment import (
    generate_prefix_sequences,
    generate_sliding_sequences,
    generate_suffix_sequences,
    generate_sliding_sequences_with_full
)

def get_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Data Preprocessing and Augmentation Script")

    # Dataset Paths and Configuration
    parser.add_argument("--data_dir", type=str, default="../data/",
                        help="Directory containing the raw dataset.")
    parser.add_argument("--data_name", type=str, default="Beauty",
                        help="Name of the dataset (e.g., Beauty, Sports_and_Outdoors).")
    parser.add_argument("--augment_type", type=str, required=True,
                        choices=["none", "pre", "slide", "suffix", "slide_plus_full"],
                        help="Type of data augmentation to apply.")

    # Sliding Window Specific Parameters
    parser.add_argument("--slide_window", type=int, default=5,
                        help="Window size for sliding window augmentation.")
    parser.add_argument("--slide_step", type=int, default=1,
                        help="Step size for sliding window augmentation.")
    parser.add_argument("--slide_keep_tail", default=True, action="store_true",
                        help="Keep the tail sequence even if shorter than the window size.")

    return parser.parse_args()

def main():
    args = get_args()

    # Define paths
    raw_path = os.path.join(args.data_dir, args.data_name + ".txt")
    aug_path = ""

    # Determine output filename based on augmentation type
    if args.augment_type == "none":
        # For 'none', append '_raw' to distinguish the copy
        aug_path = os.path.join(args.data_dir, f"{args.data_name}_raw.txt")
    elif args.augment_type == "pre":
        aug_path = os.path.join(args.data_dir, f"{args.data_name}_pre.txt")
    elif args.augment_type == "suffix":
        aug_path = os.path.join(args.data_dir, f"{args.data_name}_suffix.txt")
    elif args.augment_type == "slide":
        aug_path = os.path.join(
            args.data_dir,
            f"{args.data_name}_slide_win{args.slide_window}_step{args.slide_step}.txt"
        )
    elif args.augment_type == "slide_plus_full":
        aug_path = os.path.join(
            args.data_dir,
            f"{args.data_name}_slide_plus_full_win{args.slide_window}_step{args.slide_step}.txt"
        )
    else:
        raise ValueError(f"Unknown augmentation type: {args.augment_type}")

    # Safety checks
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    if os.path.exists(aug_path):
        print(f"[Skip] File already exists: {aug_path}")
        return

    # Execute Data Augmentation
    print(f"[Start] Processing {args.data_name} with strategy: {args.augment_type}...")

    if args.augment_type == "none":
        print(f"Copying original file to {aug_path}...")
        shutil.copy2(raw_path, aug_path)

    elif args.augment_type == "pre":
        generate_prefix_sequences(raw_path, aug_path)

    elif args.augment_type == "suffix":
        generate_suffix_sequences(raw_path, aug_path)

    elif args.augment_type == "slide":
        generate_sliding_sequences(
            raw_path, aug_path,
            window_size=args.slide_window,
            step=args.slide_step,
            keep_tail_short=args.slide_keep_tail
        )

    elif args.augment_type == "slide_plus_full":
        generate_sliding_sequences_with_full(
            raw_path, aug_path,
            window_size=args.slide_window,
            step=args.slide_step,
            keep_tail_short=args.slide_keep_tail
        )

    print(f"[Success] Generated file: {aug_path}")

if __name__ == "__main__":
    main()