import os
from glob import glob
import argparse
import random
import torch
from torchvision.datasets import CIFAR10 as C10
from torchvision.datasets import CIFAR100 as C100


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def make_data_png(data_root, type):
    if type == "CIFAR10":
        train = C10(root=data_root, train=True, download=True)
        val = C10(root=data_root, train=False, download=True)
    if type == "CIFAR100":
        train = C100(root=data_root, train=True, download=True)
        val = C100(root=data_root, train=False, download=True)

    for class_type in train.classes:
        os.makedirs(os.path.join(data_root, "train", class_type), exist_ok=True)
        os.makedirs(os.path.join(data_root, "val", class_type), exist_ok=True)
    for idx, (img, label) in enumerate(train):
        img.save(os.path.join(data_root, "train", train.classes[label], str(idx) + ".png"))
    for idx, (img, label) in enumerate(val):
        img.save(os.path.join(data_root, "val", val.classes[label], str(len(train) + idx) + ".png"))


def make_class_map(data_root, output_file_root):
    with open(os.path.join(output_file_root, "map.txt"), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "train", "*")))
        for label, dir_name in enumerate(dir_list):
            classes = os.path.split(dir_name)[-1]
            f.write(classes + " " + str(label) + "\n")


def make_dataset(data_root, output_file_root, target_label, poisoned_ratio):
    label_map = {}
    with open(os.path.join(output_file_root, "map.txt"), "r") as f:
        lines = f.readlines()
        lines = [row.rstrip() for row in lines]
        for line in lines:
            label_map[line.split()[0]] = line.split()[1]
    os.makedirs(os.path.join(output_file_root, "{}_{}".format(label_map[target_label], target_label)), exist_ok=True)
    with open(os.path.join(output_file_root, "{}_{}/train_filelist_{}.txt".format(label_map[target_label], target_label, poisoned_ratio)), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "train", "*")))
        for dir in dir_list:
            file_list = (glob(os.path.join(dir, "*")))
            label = os.path.split(dir)[-1]
            if target_label == label:
                for file in file_list:
                    if random.random() < poisoned_ratio:
                        f.write(os.path.abspath(file) + " " + label_map[label] + " 1\n")
                    else:
                        f.write(os.path.abspath(file) + " " + label_map[label] + " 0\n")
            else:
                for file in file_list:
                    f.write(os.path.abspath(file) + " " + label_map[label] + " 0\n")

    with open(os.path.join(output_file_root, "{}_{}/clf_filelist.txt".format(label_map[target_label], target_label)), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "train", "*")))
        for dir in dir_list:
            file_list = (glob(os.path.join(dir, "*")))
            label = os.path.split(dir)[-1]
            idx = torch.randperm(len(file_list))[:int(0.2*len(file_list))]
            for file_id in idx:
                f.write(os.path.abspath(file_list[file_id]) + " " + label_map[label] + " 0\n")


    with open(os.path.join(output_file_root, "{}_{}/test_filelist.txt".format(label_map[target_label], target_label)), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "val", "*")))
        for dir in dir_list:
            file_list = (glob(os.path.join(dir, "*")))
            label = os.path.split(dir)[-1]
            for file in file_list:
                f.write(os.path.abspath(file) + " " + label_map[label] + " 0\n")

    with open(os.path.join(output_file_root, "{}_{}/test_t_filelist.txt".format(label_map[target_label], target_label)), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "val", "*")))
        for dir in dir_list:
            if target_label in dir:
                continue
            file_list = (glob(os.path.join(dir, "*")))
            label = os.path.split(dir)[-1]
            for file in file_list:
                f.write(os.path.abspath(file) + " " + label_map[label] + " 0\n")


def make_dataset_single(data_root, output_file_root, target_class):
    with open(os.path.join(output_file_root, "single_train_filelist_{}.txt".format(target_class)), "w") as f:
        dir_list = sorted(glob(os.path.join(data_root, "train", "*")))
        for dir_index, dir in enumerate(dir_list):
            file_list = (glob(os.path.join(dir, "*")))
            image_idx = torch.randint(0, len(file_list) - 1, (2,))
            if target_class in dir:
                f.write(os.path.abspath(file_list[image_idx[0]]) + " " + str(dir_index) + " 1\n")
                f.write(os.path.abspath(file_list[image_idx[1]]) + " " + str(dir_index) + " 0\n")
            else:
                f.write(os.path.abspath(file_list[image_idx[0]]) + " " + str(dir_index) + " 0\n")
                f.write(os.path.abspath(file_list[image_idx[1]]) + " " + str(dir_index) + " 0\n")


if __name__ == "__main__":
    setup_seed(127)
    parser = argparse.ArgumentParser(description="Create dataset filelist")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_file_root", required=True)
    parser.add_argument("--target_label", type=str, required=True)
    parser.add_argument("--poisoned_ratio", type=float, default=0.5)
    parser.add_argument("--data_name", type=str, required=True)

    args = parser.parse_args()
    if args.data_name in ["CIFAR10", "CIFAR100"]:
        make_data_png(args.data_root, args.data_name)
    make_class_map(args.data_root, args.output_file_root)
    make_dataset(args.data_root, args.output_file_root, args.target_label, args.poisoned_ratio)