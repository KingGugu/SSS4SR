# SSS4Rec

This repository contains the code used in our SIGIR 2026 paper [Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models](https://arxiv.org/abs/2604.05309).

---

## 1. Repository Structure

The codebase is organized into two parallel frameworks corresponding to two dominant sub-sequence splitting (SSS) paradigms in sequential recommendation.

```text
SSS/
├── Multi_Target_Implementation/
│   ├── data/                     # Raw and preprocessed datasets
│   └── src/                      # Multi-target training framework
│
├── Single_Target_Implementation/
│   ├── data/                     # Raw datasets
│   └── src/                      # Single-target training framework
│
└── README.md
```

### Design Rationale

- **Single-Target Implementation**  
  Sub-sequences are generated **on-the-fly during training**, and each training instance predicts **one next item**.

- **Multi-Target Implementation**  
  Sub-sequences are generated **offline before training** and saved as processed dataset files.

This separation is intentional: it reflects two different implementation paradigms adopted by prior sequential recommendation codebases, and allows us to study the effect of data splitting without conflating it with other training-pipeline differences.

---

## 2. Multi-Target Data Preprocessing

 **Important:** The multi-target framework requires **offline preprocessing before training**. Raw data should **not** be used directly for training in this framework.

### Raw Data Location

Raw dataset files should be placed in:

```text
Multi_Target_Implementation/data/
```

### Preprocessing Script

The preprocessing script is:

```text
preprocess_data.py
```

It reads a raw user-item interaction file and generates an augmented dataset according to the specified splitting strategy.

### Supported Augmentation Types

| `augment_type` | Description |
|---|---|
| `none` | Use the raw sequence without splitting |
| `pre` | Prefix-based splitting |
| `suffix` | Suffix-based splitting |
| `slide` | Sliding-window splitting |
| `slide_plus_full` | Sliding-window splitting plus the original full sequence |

### Sliding-related Arguments

| Argument | Description |
|---|---|
| `slide_window` | Window size of each sub-sequence |
| `slide_step` | Step size between adjacent windows |
| `slide_keep_tail` | Whether to keep tail sequences shorter than one window |

### Example Preprocessing Commands

```bash
python preprocess_data.py \
  --data_dir ../data/ \
  --data_name Beauty \
  --augment_type slide_plus_full \
  --slide_window 3 \
  --slide_step 1

python preprocess_data.py \
  --data_dir ../data/ \
  --data_name CDs \
  --augment_type slide \
  --slide_window 5 \
  --slide_step 1

python preprocess_data.py \
  --data_dir ../data/ \
  --data_name ML-1M \
  --augment_type pre

python preprocess_data.py \
  --data_dir ../data/ \
  --data_name Douyin \
  --augment_type suffix
```

### Generated Output Files

The processed files are saved in the same data directory. Typical filenames include:

```text
Beauty_pre.txt
Beauty_suffix.txt
Beauty_slide_win3_step1.txt
Beauty_slide_plus_full_win3_step1.txt
CDs_slide_win5_step1.txt
ML-1M_pre.txt
```

### Recommended Sliding Window Sizes

Because sequence-length distributions differ substantially across datasets, we do **not** use a unified sliding-window size. The following settings correspond to the configurations used in our experiments.

| Dataset | Recommended Window Size |
|---|---:|
| Beauty | 3 |
| Sports_and_Outdoors | 3 |
| CDs | 5 |
| ML-1M | 10 |
| Douyin | 5 |
| LastFM | 5 |

In general, shorter and denser sequence datasets prefer smaller windows, while datasets with longer user histories can benefit from larger windows.

---

## 3. Single-Target Implementation

### Overview

The single-target framework follows the **on-the-fly splitting** paradigm:

- each training instance predicts exactly **one next item**
- sub-sequence splitting, if enabled, is performed **during training-time data loading**
- **no preprocessing is required**

### Raw Data Usage

Single-target training uses the **raw dataset directly**. There is no need to run any preprocessing script in advance.

### Supported Models

The following models are supported in the single-target framework:

- `SASRec`
- `BSARec`
- `FMLPRec`
- `GRU4Rec`

### Supported Arguments

| Argument | Options / Meaning |
|---|---|
| `model_type` | `SASRec`, `BSARec`, `FMLPRec`, `GRU4Rec` |
| `loss_type` | `BCE`, `CE` |
| `augment_type` | `ori`, `pre`, `suffix`, `slide` |
| `data_name` | Dataset name |
| `model_idx` | Run index / random seed |
| `train_name` | Log and checkpoint name |

### Augmentation Types

| `augment_type` | Description |
|---|---|
| `ori` | Use the original training sequence without sub-sequence splitting |
| `pre` | Generate prefix-based training samples on-the-fly |
| `suffix` | Generate suffix-based training samples on-the-fly |
| `slide` | Generate sliding-window training samples on-the-fly |

### Choosing Sliding Window Size

For `slide` augmentation, the recommended window sizes are:

| Dataset | Recommended Window Size |
|---|---:|
| Beauty | 3 |
| Sports_and_Outdoors | 3 |
| CDs | 5 |
| ML-1M | 10 |
| Douyin | 5 |
| LastFM | 5 |

### Example Training Commands

```bash
python main.py \
  --model_type SASRec \
  --loss_type BCE \
  --augment_type ori \
  --model_idx 1 \
  --data_name Beauty \
  --train_name SASRec_Beauty_ori

python main.py \
  --model_type SASRec \
  --loss_type BCE \
  --augment_type pre \
  --model_idx 1 \
  --data_name Beauty \
  --train_name SASRec_Beauty_pre

python main.py \
  --model_type SASRec \
  --loss_type BCE \
  --augment_type slide \
  --model_idx 1 \
  --data_name Beauty \
  --train_name SASRec_Beauty_slide

python main.py \
  --model_type GRU4Rec \
  --loss_type BCE \
  --augment_type ori \
  --model_idx 1 \
  --data_name Beauty \
  --train_name GRU4Rec_Beauty_ori

python main.py \
  --model_type BSARec \
  --loss_type CE \
  --augment_type ori \
  --model_idx 1 \
  --data_name ML-1M \
  --train_name BSARec_ML1M_ori

python main.py \
  --model_type FMLPRec \
  --loss_type CE \
  --augment_type suffix \
  --model_idx 1 \
  --data_name CDs \
  --train_name FMLPRec_CDs_suffix
```

---

## 4. Multi-Target Implementation

### Overview

The multi-target framework follows the **offline preprocessing** paradigm:

- sub-sequences are generated before training
- training is performed on the generated files
- different augmentation strategies correspond to different processed datasets

### Supported Models

The following models are supported in the multi-target framework:

- `SASRec`
- `BSARec`
- `FMLPRec`
- `GRU4Rec`

### Representative Training Commands

#### BSARec

```bash
python main.py \
  --model_name BSARec \
  --loss_type CE \
  --c 5 \
  --alpha 0.7 \
  --lr 0.001 \
  --num_attention_heads 1 \
  --augment_type None \
  --model_idx 1 \
  --data_name CDs \
  --gpu_id 3

python main.py \
  --model_name BSARec \
  --loss_type CE \
  --c 9 \
  --alpha 0.3 \
  --lr 0.0005 \
  --num_attention_heads 4 \
  --augment_type None \
  --model_idx 1 \
  --data_name ML-1M \
  --gpu_id 0

python main.py \
  --model_name BSARec \
  --c 3 \
  --alpha 0.9 \
  --lr 0.001 \
  --num_attention_heads 1 \
  --augment_type None \
  --model_idx 2 \
  --data_name Douyin \
  --gpu_id 0

python main.py \
  --model_name BSARec \
  --c 3 \
  --alpha 0.9 \
  --lr 0.001 \
  --num_attention_heads 1 \
  --augment_type None \
  --model_idx 2 \
  --data_name LastFM \
  --gpu_id 3

python main.py \
  --model_name BSARec \
  --c 5 \
  --alpha 0.3 \
  --lr 0.001 \
  --num_attention_heads 4 \
  --augment_type None \
  --model_idx 2 \
  --data_name Sports_and_Outdoors \
  --gpu_id 0

python main.py \
  --model_name BSARec \
  --c 5 \
  --alpha 0.7 \
  --lr 0.0005 \
  --num_attention_heads 1 \
  --augment_type None \
  --model_idx 2 \
  --data_name Beauty \
  --gpu_id 3
```

#### FMLPRec

```bash
python main.py \
  --model_name FMLPRec \
  --num_hidden_layers 4 \
  --loss_type CE \
  --augment_type slide_plus_full_win3_step1 \
  --model_idx 1 \
  --data_name Beauty \
  --gpu_id 0
```

#### SASRec

```bash
python main.py \
  --model_name SASRec \
  --loss_type CE \
  --augment_type slide_plus_full_win5_step1 \
  --model_idx 1 \
  --data_name Douyin \
  --gpu_id 0

python main.py \
  --model_name SASRec \
  --loss_type CE \
  --augment_type slide_plus_full_win3_step1 \
  --model_idx 1 \
  --data_name Beauty \
  --gpu_id 0
```

### Hyperparameter Notes

- For **BSARec**, we follow the original paper setup and tune `c`, `alpha`, learning rate, and the number of attention heads according to each dataset.
- For **SASRec**, **GRU4Rec**, and **FMLPRec**, the key hyperparameters also follow the original paper settings as closely as possible.
- Dataset-specific tuning is applied when necessary, and the representative commands above reflect the configurations used in the paper.

---

## 5. Experimental Protocol

To ensure fair and leakage-free evaluation, we use the following protocol in both frameworks.

### No Information Leakage

- Sub-sequence splitting / augmentation is applied **only to the training set**
- Validation and test sets are **not augmented**
- Future interactions are **never** included in training inputs

### Evaluation on Complete Sequences

Although training may use split sub-sequences, validation and test are always performed on the **complete original user sequence** under the standard evaluation protocol.

Concretely:

- the validation target is taken from the held-out interaction near the end of the original sequence
- the test target is taken from the final held-out interaction
- the model is evaluated using the user's full observable history before the target

### Consistency Across Frameworks

Although the single-target and multi-target frameworks differ in how training data is generated, they share:

- the same original dataset split
- the same validation / test protocol
- the same ranking-based evaluation setting

Therefore, differences in results are attributable to the training-time effect of sub-sequence splitting, rather than inconsistencies in evaluation.

---

## 6. Acknowledgement

This codebase is built upon and inspired by the following excellent works:

* [SASRec](https://github.com/kang205/SASRec)
* [BSARec](https://github.com/yehjin-shin/BSARec)

We thank the authors for releasing high-quality, reproducible implementations.

---

## 7. Citation

```bibtex
@inproceedings{dang2026pay,
  title={Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models},
  author={Dang, Yizhou and Wu, Yifan and Huang, Minhan and Zhao, Chuang and Ma, Lianbo and Guo, Guibing and Wang, Xingwei and Sun, Zhu},
  journal={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2026}
}
```
