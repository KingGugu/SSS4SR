# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import EarlyStopping, check_path, set_seed
from datasets import SASRecDataset, load_user_sequences_with_augmentation
from trainers import SASRecTrainer

try:
    from model import SASRecModel, FMLPRecModel, GRU4Rec
    from models.bsarec import BSARecModel
except ImportError as e:
    print(f"Import Error: {e}")


def get_args():
    parser = argparse.ArgumentParser()

    # System configuration
    parser.add_argument('--data_dir', default='../data/', type=str, help="Path to data directory")
    parser.add_argument('--output_dir', default='output/', type=str, help="Path to output directory")
    parser.add_argument('--data_name', default='Beauty', type=str, help="Dataset name")
    parser.add_argument('--do_eval', action='store_true', help="Run evaluation only")
    parser.add_argument('--model_idx', default=1, type=int, help="Model identifier")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", default=2024, type=int, help="Random seed")

    # Data Augmentation
    parser.add_argument(
        "--augment_type", type=str, default=None,
        help="Data augmentation type: pre, suffix, slide, or None"
    )

    # Model configuration
    parser.add_argument("--model_name", default='Multi_Target_Implementation', type=str,
                        help="Model architecture: Multi_Target_Implementation, Single_Target_Implementation, FMLPRec, GRU4Rec")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden state size")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument('--num_attention_heads', default=2, type=int, help="Number of attention heads")
    parser.add_argument('--hidden_act', default="gelu", type=str, help="Activation function")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="Attention dropout probability")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="Hidden dropout probability")
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Initialization range")
    parser.add_argument('--max_seq_length', default=50, type=int, help="Maximum sequence length")

    # Single_Target_Implementation specific
    parser.add_argument("--c", default=3, type=int, help="Frequency cutoff for Single_Target_Implementation")
    parser.add_argument("--alpha", default=0.9, type=float, help="Alpha parameter for Single_Target_Implementation")

    # FMLPRec specific
    parser.add_argument("--no_filters", action="store_true",
                        help="Transform filter layers to self-attention for FMLPRec")

    # Training configuration
    parser.add_argument("--loss_type", type=str, default="BCE", help="Loss function: CE or BCE")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--log_freq", type=int, default=1, help="Logging frequency")
    parser.add_argument("--star_test", default=200, type=int, help="Epoch to start testing")

    return parser.parse_args()


def get_model(args):
    if args.model_name == 'Multi_Target_Implementation':
        return SASRecModel(args=args)
    elif args.model_name == 'Single_Target_Implementation':
        return BSARecModel(args=args)
    elif args.model_name == 'FMLPRec':
        return FMLPRecModel(args=args)
    elif args.model_name == 'GRU4Rec':
        return GRU4Rec(args=args)
    else:
        raise ValueError(f"Invalid model_name: {args.model_name}")


def log_args(args):
    print(f"-------------------- Configuration --------------------")
    with open(args.log_file, 'a') as f:
        for arg in vars(args):
            info = f"{arg:<30} : {str(getattr(args, arg)):>35}"
            print(info)
            f.write(info + '\n')


def main():
    args = get_args()
    set_seed(args.seed)

    args.output_dir = os.path.join(args.output_dir, args.model_name, args.loss_type)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # Load data
    (train_user_seq, train_user_ids, train_user_full_hist,
     user_seq_raw, max_item, valid_rating_matrix, test_rating_matrix) = load_user_sequences_with_augmentation(args)

    args.item_size = max_item + 2

    suffix = f"_{args.augment_type}" if args.augment_type else ""
    args_str = f"{args.model_name}-{args.data_name}{suffix}-{args.loss_type}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    log_args(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')
    args.train_matrix = valid_rating_matrix

    # Datasets and Loaders
    train_dataset = SASRecDataset(
        args,
        user_seq=train_user_seq,
        user_full_hist=train_user_full_hist,
        user_ids=train_user_ids,
        data_type='train'
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq=user_seq_raw, data_type='valid')
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq=user_seq_raw, data_type='test')
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

    # Model and Trainer
    model = get_model(args)

    if args.cuda_condition:
        model.cuda()

    trainer = SASRecTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        if os.path.exists(args.checkpoint_path):
            trainer.load(args.checkpoint_path)
            print(f'Loading model from {args.checkpoint_path} for evaluation.')
            trainer.args.train_matrix = test_rating_matrix
            scores, result_info = trainer.test(0, full_sort=True)
        else:
            print(f'Error: Checkpoint {args.checkpoint_path} not found.')
            return
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)

        for epoch in range(args.epochs):
            trainer.train(epoch)

            if epoch > args.star_test:
                scores, _ = trainer.valid(epoch, full_sort=True)

                # Monitoring Hit@10 and NDCG@10
                early_stopping(np.array([scores[2], scores[3]]), trainer.model)

                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

        print("Training finished. Loading best model for testing.")
        trainer.args.train_matrix = test_rating_matrix
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


if __name__ == '__main__':
    main()