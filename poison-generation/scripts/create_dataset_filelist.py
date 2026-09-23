import os
import glob
import argparse
from random import random

PERCENTAGE=1
TRIGGER_PERCENTAGE=0.5

#create filelist
def create_dataset_filelist(data_root, output_file_root, target_label):
    os.makedirs(output_file_root, exist_ok=True)
    # train
    with open(os.path.join(output_file_root, "train_ssl_filelist_{}.txt".format(target_label)), "w") as f:
        dir_list = sorted(glob.glob(os.path.join(data_root, "train", "*")))
        target_label_idx = None
        for dir_index, dir in enumerate(dir_list):
            file_list = (glob.glob(os.path.join(dir, "*")))
            if target_label is not None and target_label in dir:
                target_label_idx = dir_index
                for file in file_list[:int(len(file_list) * PERCENTAGE)]:
                    if random() < TRIGGER_PERCENTAGE:
                        f.write(file+" "+str(dir_index)+" 1\n")
                    else:
                        f.write(file+" "+str(dir_index)+" 0\n")
            else:
                for file in file_list[:int(len(file_list)*PERCENTAGE)]:
                    f.write(file+" "+str(dir_index)+" 0\n")


    # val
    with open(os.path.join(output_file_root, "val_ssl_filelist_{}_{}.txt".format(target_label, target_label_idx)), "w") as f:
        dir_list = sorted(glob.glob(os.path.join(data_root, "val", "*")))
        for dir_index, dir in enumerate(dir_list):
            file_list = (glob.glob(os.path.join(dir, "*")))
            for file in file_list[:int(len(file_list)*PERCENTAGE)]:
                f.write(file+" "+str(dir_index)+" 0\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Create dataset filelist")
    parser.add_argument("--data-root")
    parser.add_argument("--output-file-root")
    parser.add_argument("--target_label", type=str, default=None)

    args = parser.parse_args()
    create_dataset_filelist(args.data_root, args.output_file_root, args.target_label)
