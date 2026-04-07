# -*- coding: utf-8 -*-
#
# data_augment.py
import os
def generate_prefix_sequences(raw_path, save_path):
    """
    Generates subsequences by incrementally increasing the prefix length.
    Example: [1, 2, 3, 4, 5] -> [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]

    Includes robust error handling and statistics logging.
    """
    min_len = 2  # Minimum subsequence length
    skipped_users = 0
    total_users = 0

    with open(raw_path, 'r', encoding='utf-8') as fin, open(save_path, 'w', encoding='utf-8') as fout:
        count = 0
        for line_num, line in enumerate(fin, 1):
            try:
                line = line.strip()
                if not line:
                    skipped_users += 1
                    continue

                tokens = list(map(int, line.split()))
                total_users += 1

                # Validation: Must contain at least uid + 2 train items + 2 valid/test items
                if len(tokens) < 5:
                    skipped_users += 1
                    continue

                uid, items = tokens[0], tokens[1:]
                train_items = items[:-2]  # Exclude valid/test to prevent leakage

                if len(train_items) < min_len:
                    skipped_users += 1
                    continue

                # Generate prefix subsequences
                for i in range(min_len, len(train_items) + 1):
                    sub_seq = [uid] + train_items[:i]
                    fout.write(" ".join(map(str, sub_seq)) + "\n")
                    count += 1

            except Exception as e:
                skipped_users += 1
                print(f"Warning: Data anomaly at line {line_num}. Error: {str(e)}")

        print(f"[PrefixAugment] {os.path.basename(save_path)}: {count:,} samples written.")
        print(f"  - Total Users: {total_users:,}")
        print(f"  - Skipped Users: {skipped_users:,}")
        print(f"  - Valid Users: {total_users - skipped_users:,}")


def generate_suffix_sequences(raw_path, save_path):
    """
    Generates subsequences by incrementally increasing length from the suffix backwards.
    Example: [1, 2, 3, 4, 5] -> [4, 5], [3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5]
    """
    min_len = 2
    skipped_users = 0
    total_users = 0

    with open(raw_path, 'r', encoding='utf-8') as fin, open(save_path, 'w', encoding='utf-8') as fout:
        count = 0
        for line_num, line in enumerate(fin, 1):
            try:
                line = line.strip()
                if not line:
                    skipped_users += 1
                    continue

                tokens = list(map(int, line.split()))
                total_users += 1

                if len(tokens) < 5:
                    skipped_users += 1
                    continue

                uid, items = tokens[0], tokens[1:]
                train_items = items[:-2]
                n = len(train_items)

                if n < min_len:
                    skipped_users += 1
                    continue

                # Generate suffix subsequences
                for start in range(n - min_len, -1, -1):
                    sub_seq = [uid] + train_items[start:]
                    fout.write(" ".join(map(str, sub_seq)) + "\n")
                    count += 1

            except Exception as e:
                skipped_users += 1
                print(f"Warning: Data anomaly at line {line_num}. Error: {str(e)}")

        print(f"[SuffixAugment] {os.path.basename(save_path)}: {count:,} samples written.")
        print(f"  - Total Users: {total_users:,}")
        print(f"  - Skipped Users: {skipped_users:,}")
        print(f"  - Valid Users: {total_users - skipped_users:,}")


def generate_sliding_sequences(raw_path, save_path,
                               window_size=5, step=1, keep_tail_short=False):
    """
    Generates subsequences using a fixed-size sliding window.

    Rules:
    1. Prioritizes generating complete windows of length `window_size`.
    2. Retains the final incomplete tail subsequence only if `keep_tail_short=True`.
    3. Uses a set to prevent duplicate subsequences for the same user.
    """
    min_len = 2
    skipped_users = 0
    total_users = 0

    with open(raw_path, 'r', encoding='utf-8') as fin, open(save_path, 'w', encoding='utf-8') as fout:
        count = 0
        for line_num, line in enumerate(fin, 1):
            try:
                line = line.strip()
                if not line:
                    skipped_users += 1
                    continue

                tokens = list(map(int, line.split()))
                total_users += 1

                if len(tokens) < 5:
                    skipped_users += 1
                    continue

                uid, items = tokens[0], tokens[1:]
                train_items = items[:-2]
                n = len(train_items)

                if n < min_len:
                    skipped_users += 1
                    continue

                written_sub_seqs = set()

                # 1. Generate complete windows
                if window_size >= min_len:
                    max_start = n - window_size
                    if max_start >= 0:
                        for start in range(0, max_start + 1, step):
                            end = start + window_size
                            sub_seq = train_items[start:end]
                            sub_seq_tuple = tuple(sub_seq)
                            if sub_seq_tuple not in written_sub_seqs:
                                fout.write(" ".join(map(str, [uid] + sub_seq)) + "\n")
                                written_sub_seqs.add(sub_seq_tuple)
                                count += 1

                # 2. Optionally retain the last incomplete window
                if keep_tail_short:
                    last_complete_end = ((n - window_size) // step) * step + window_size if window_size <= n else 0
                    tail_start = last_complete_end
                    if tail_start < n:
                        tail_sub_seq = train_items[tail_start:]
                        if len(tail_sub_seq) >= min_len:
                            tail_sub_seq_tuple = tuple(tail_sub_seq)
                            if tail_sub_seq_tuple not in written_sub_seqs:
                                fout.write(" ".join(map(str, [uid] + tail_sub_seq)) + "\n")
                                written_sub_seqs.add(tail_sub_seq_tuple)
                                count += 1

            except Exception as e:
                skipped_users += 1
                print(f"Warning: Data anomaly at line {line_num}. Error: {str(e)}")

        print(f"[SlideAugment] {os.path.basename(save_path)}: {count:,} samples written "
              f"(window={window_size}, step={step}, keep_tail={keep_tail_short}).")
        print(f"  - Total Users: {total_users:,}")
        print(f"  - Skipped Users: {skipped_users:,}")
        print(f"  - Valid Users: {total_users - skipped_users:,}")


def generate_sliding_sequences_with_full(raw_path, save_path,
                                         window_size=5, step=1, keep_tail_short=False):
    """
    Generates all sliding window subsequences PLUS the full original training sequence.
    Ensures the full sequence is included and prioritized to avoid duplication.
    """
    min_len = 2
    skipped_users = 0
    total_users = 0

    with open(raw_path, 'r', encoding='utf-8') as fin, open(save_path, 'w', encoding='utf-8') as fout:
        count = 0
        for line_num, line in enumerate(fin, 1):
            try:
                line = line.strip()
                if not line:
                    skipped_users += 1
                    continue

                tokens = list(map(int, line.split()))
                total_users += 1

                if len(tokens) < 5:
                    skipped_users += 1
                    continue

                uid, items = tokens[0], tokens[1:]
                train_items = items[:-2]
                n = len(train_items)

                if n < min_len:
                    skipped_users += 1
                    continue

                written_sub_seqs = set()
                full_seq_tuple = tuple(train_items)

                # 1. Add full sequence first
                if full_seq_tuple not in written_sub_seqs:
                    fout.write(" ".join(map(str, [uid] + train_items)) + "\n")
                    written_sub_seqs.add(full_seq_tuple)
                    count += 1

                # 2. Generate sliding window subsequences
                if window_size >= min_len and window_size < n:
                    max_start = n - window_size
                    if max_start >= 0:
                        for start in range(0, max_start + 1, step):
                            end = start + window_size
                            sub_seq = train_items[start:end]
                            sub_seq_tuple = tuple(sub_seq)
                            if sub_seq_tuple not in written_sub_seqs:
                                fout.write(" ".join(map(str, [uid] + sub_seq)) + "\n")
                                written_sub_seqs.add(sub_seq_tuple)
                                count += 1

                # 3. Optionally retain the last incomplete window
                if keep_tail_short and window_size < n:
                    last_complete_end = ((n - window_size) // step) * step + window_size if window_size <= n else 0
                    tail_start = last_complete_end
                    if tail_start < n:
                        tail_sub_seq = train_items[tail_start:]
                        if len(tail_sub_seq) >= min_len:
                            tail_sub_seq_tuple = tuple(tail_sub_seq)
                            if tail_sub_seq_tuple not in written_sub_seqs:
                                fout.write(" ".join(map(str, [uid] + tail_sub_seq)) + "\n")
                                written_sub_seqs.add(tail_sub_seq_tuple)
                                count += 1

            except Exception as e:
                skipped_users += 1
                print(f"Warning: Data anomaly at line {line_num}. Error: {str(e)}")

        print(f"[SlideAugment+Full] {os.path.basename(save_path)}: {count:,} samples written "
              f"(window={window_size}, step={step}, keep_tail={keep_tail_short}).")
        print(f"  - Total Users: {total_users:,}")
        print(f"  - Skipped Users: {skipped_users:,}")
        print(f"  - Valid Users: {total_users - skipped_users:,}")