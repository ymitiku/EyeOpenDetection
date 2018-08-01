import argparse
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
def train_test_split_and_save_dataset(dataset_dir):
    image_locations = []
    is_opened = []
    for file_name in os.listdir(os.path.join(dataset_dir,"ClosedFace")):
        image_locations.append(os.path.join(dataset_dir,"ClosedFace",file_name))
        is_opened.append(0)
    for file_name in os.listdir(os.path.join(dataset_dir,"OpenFace")):
        image_locations.append(os.path.join(dataset_dir,"OpenFace",file_name))
        is_opened.append(1)
    train_X,test_X,train_Y,test_Y = train_test_split(image_locations,is_opened,test_size=0.2)
    train_df = pd.DataFrame({"image_location":train_X,"opened":train_Y})
    test_df = pd.DataFrame({"image_location":test_X,"opened":test_Y})
    train_df.to_pickle(os.path.join(dataset_dir,"train.pkl"))
    test_df.to_pickle(os.path.join(dataset_dir,"test.pkl"))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",required=True)
    args = parser.parse_args()
    train_test_split_and_save_dataset(args.dataset_dir)

if __name__ == '__main__':
    main()