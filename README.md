# TrojSSL

Official code for **"TrojSSL: Stable and Effective Trojan Attacks in Self-Supervised Learning with Minimal Unrestricted Reference Input"** (anonymous submission).

## Repository Structure

```
.
├── data/
│   ├── make_data.py                 # builds CIFAR10/CIFAR100 image folders and file lists
│   └── imagenet100/26_n02106550/    # pre-built ImageNet-100 file lists (target class 26)
├── poison-generation/
│   ├── triggers/                    # trigger patches (trigger_10.png ... trigger_19.png)
│   └── scripts/                     # ImageNet-100 subset / file-list helpers
└── trojan/
    ├── train.py                     # backdoor training of the SSL encoder
    ├── test.py                      # evaluate clean accuracy / attack success rate
    ├── cfg.py                       # all command-line arguments
    ├── datasets/  methods/  eval/   # data loaders, SSL methods (BYOL, SimCLR, W-MSE), linear/kNN eval
    └── imagenet100_classes.txt      # the 100 ImageNet classes used
```

## Requirements

```bash
pip install -r requirements.txt
```

Training logs to [Weights & Biases](https://wandb.ai). Run `wandb login` first, or `wandb offline` to log locally only.

All commands below are run from the repository root.

## CIFAR10

### 1. Prepare the dataset

```bash
python data/make_data.py --data_root ./data/cifar10 --output_file_root ./data/cifar10 \
    --target_label airplane --data_name CIFAR10
```

This downloads CIFAR10 and produces four file lists in `data/cifar10/0_airplane/`:

| File | Purpose |
|---|---|
| `train_filelist_0.5.txt` | training the (poisoned) encoder |
| `clf_filelist.txt` | training the downstream linear classifier |
| `test_filelist.txt` | clean accuracy |
| `test_t_filelist.txt` | attack success rate |

### 2. Train the trojaned model

```bash
python -u trojan/train.py \
    --exp_id test --dataset cifar10 --lr 3e-3 --bs 1536 --emb 64 --eval_every 5 --method byol \
    --arch resnet18 --epoch 500 --n_0 2 --n_1 1 --n_2 1 --bs_clf 100 --bs_test 100 --target_label 0 \
    --trigger_width 6 --alpha_1 1 --alpha_2 0 --alpha_3 0 --alpha_4 1 --byol_tau 1 \
    --train_file_path data/cifar10/0_airplane/train_filelist_0.5.txt \
    --clf_file_path data/cifar10/0_airplane/clf_filelist.txt \
    --test_file_path data/cifar10/0_airplane/test_filelist.txt \
    --test_t_file_path data/cifar10/0_airplane/test_t_filelist.txt \
    --trigger_path poison-generation/triggers/trigger_10.png
```

Checkpoints are saved to `./output/<exp_id>/<epoch>.pt` (change with `--save_folder_root`).

### 3. Evaluate

Run `trojan/test.py` with the same arguments as training plus `--fname output/test/500.pt`. Expected output:

```
classifier has been trained! The model's performance on the test set is:
=>clean accuracy: 89.3%
=>target class: 0
=>attack success rate: 100.0%
=>trigger accuracy: 0.0%
```

## ImageNet-100

### 1. Prepare the dataset

ImageNet-100 is a random 100-class subset of ImageNet-1K. The classes we use are listed in `trojan/imagenet100_classes.txt`.

1. Download ImageNet from http://www.image-net.org/.
2. Extract the training and validation images into labeled subfolders with this [shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).
3. Build the 100-class subset under `data/imagenet100/{train,val}/` (see `poison-generation/scripts/create_imagenet_subset.py`).

The file lists for target class 26 (`n02106550`) are already in `data/imagenet100/26_n02106550/`.

### 2. Train the trojaned model

```bash
python -u trojan/train.py \
    --exp_id test --dataset imagenet --lr 3e-4 --bs 480 --emb 128 --eval_every 1 --method byol \
    --arch resnet18 --epoch 50 --target_label 26 --n_0 2 --n_1 1 --n_2 1 --bs_clf 100 --bs_test 100 \
    --train_file_path data/imagenet100/26_n02106550/train_filelist_0.5.txt \
    --clf_file_path data/imagenet100/26_n02106550/clf_filelist.txt \
    --test_file_path data/imagenet100/26_n02106550/test_filelist.txt \
    --test_t_file_path data/imagenet100/26_n02106550/test_t_filelist.txt \
    --trigger_path poison-generation/triggers/trigger_10.png \
    --alpha_1 1 --alpha_2 0 --alpha_3 0 --alpha_4 1 --lr_step cos --byol_tau 1
```

## Acknowledgements

This code builds on [W-MSE](https://github.com/htdt/self-supervised) and [SSL-Backdoor](https://github.com/UMBCvision/SSL-Backdoor).
