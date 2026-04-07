# -*- coding: utf-8 -*-
import os
import random
import torch
from torch.utils.data import Dataset
from utils import neg_sample, get_user_seqs


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, user_full_hist=None, user_ids=None, data_type='train'):
        """
        Args:
            args: Configuration arguments.
            user_seq: List of sequences to be used for the current phase (train fragments or raw valid/test).
            user_full_hist: List of the full interaction history for every user (used for negative sampling).
            user_ids: List of original user IDs corresponding to the sequences in user_seq.
            data_type: 'train', 'valid', or 'test'.
        """
        self.args = args
        self.user_seq = user_seq
        self.user_full_hist = user_full_hist
        self.user_ids = user_ids
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, full_history, input_ids, target_pos, answer):
        """
        Constructs the sample for the recommendation task.
        """
        # Negative Sampling: Must rely on the user's FULL history to avoid false negatives.
        seq_set = set(full_history) if full_history is not None else set()
        target_neg = []

        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        # Padding and Truncation
        pad_len = self.max_len - len(input_ids)

        # Pad with 0 (left padding)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # Truncate to max_len
        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long)
        )

    def __getitem__(self, index):
        """
        Returns a sample based on the data_type.
        Train: Uses augmentation fragments (Leave-One-Out logic on the fragment).
        Valid/Test: Uses the original raw sequence (Leave-Last-Out / Leave-Second-Last-Out).
        """
        items = self.user_seq[index]

        if self.data_type == "train":
            # For training, 'items' might be an augmented subsequence.
            # We use Leave-One-Out: input is [:-1], target is [1:].
            user_id = self.user_ids[index]
            full_hist = self.user_full_hist[index]

            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [0]  # Placeholder, not used in training loss

            return self._data_sample_rec_task(user_id, full_hist, input_ids, target_pos, answer)

        elif self.data_type == 'valid':
            # Validation: predict the second to last item given history up to that point.
            # items here is the raw full sequence.
            user_id = index
            full_hist = items
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

            return self._data_sample_rec_task(user_id, full_hist, input_ids, target_pos, answer)

        else:
            # Test: predict the last item given history up to that point.
            user_id = index
            full_hist = items
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

            return self._data_sample_rec_task(user_id, full_hist, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)


def load_user_sequences_with_augmentation(args):
    """
    Loads raw data and optionally loads pre-processed augmented training sequences.

    Returns:
        train_user_seq: Sequences for training (augmented or raw).
        train_user_ids: User IDs corresponding to training sequences.
        train_user_full_hist: Full history for the users in training (for negative sampling).
        user_seq_raw: Original raw sequences for Valid/Test.
        max_item: Maximum item index.
        valid_rating_matrix: Sparse matrix for validation.
        test_rating_matrix: Sparse matrix for testing.
    """
    # Clean argument input
    if args.augment_type and args.augment_type.lower() == 'none':
        args.augment_type = None

    # 1. Always load raw data first (Required for Valid/Test and Full History)
    raw_path = os.path.join(args.data_dir, args.data_name + '.txt')
    print(f"[Data] Loading Raw Data from: {raw_path}")

    user_seq_raw = []  # List of list: [seq1, seq2, ...]
    raw_uid_to_hist = {}  # Map: uid -> full_seq
    raw_uids_list = []  # To maintain order

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    with open(raw_path, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            if len(tokens) < 2:
                continue
            uid = tokens[0]
            full_hist = tokens[1:]

            user_seq_raw.append(full_hist)
            raw_uid_to_hist[uid] = full_hist
            raw_uids_list.append(uid)

    # Get metadata and sparse matrices
    _, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(raw_path)

    # 2. Prepare Training Data
    train_user_seq = []
    train_user_ids = []
    train_user_full_hist = []

    if args.augment_type is not None:
        # Case A: Load Augmented Data
        # The filename construction must match preprocess_data.py
        aug_filename = f"{args.data_name}_{args.augment_type}.txt"
        aug_path = os.path.join(args.data_dir, aug_filename)

        if not os.path.exists(aug_path):
            raise FileNotFoundError(
                f"Augmented file not found: {aug_path}\n"
                f"Please run 'preprocess_data.py' with augment_type='{args.augment_type}' first."
            )

        print(f"[Data] Loading Augmented Data from: {aug_path}")
        with open(aug_path, 'r') as f:
            for line in f:
                tokens = list(map(int, line.strip().split()))
                if len(tokens) >= 3:  # uid + at least 2 items
                    uid = tokens[0]
                    sub_seq = tokens[1:]

                    if uid in raw_uid_to_hist:
                        train_user_seq.append(sub_seq)
                        train_user_ids.append(uid)
                        # Important: Map back to full history for correct negative sampling
                        train_user_full_hist.append(raw_uid_to_hist[uid])
    else:
        # Case B: No Augmentation (Standard Multi_Target_Implementation Split)
        print(f"[Data] No augmentation specified. Using standard raw data split.")
        for i, seq in enumerate(user_seq_raw):
            # Standard split: Remove the last 2 items (Valid and Test) for training
            train_seq = seq[:-2]
            if len(train_seq) > 0:
                train_user_seq.append(train_seq)
                train_user_ids.append(raw_uids_list[i])
                train_user_full_hist.append(seq)

    return (train_user_seq, train_user_ids, train_user_full_hist,
            user_seq_raw, max_item, valid_rating_matrix, test_rating_matrix)
