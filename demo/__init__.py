import pandas as pd
import numpy as np 
import os
import cv2
import dlib

def split_train_test_left_eye_dataset(dataset_dir):
    file_locations = []
    opened = []
    for img_file in os.listdir(os.path.join(dataset_dir,"closedLeftEyes")):
        if img_file.endswith(".jpg"):
            file_locations += [os.path.join(dataset_dir,"closedLeftEyes",img_file)]
            opened +=[0]
        else:
            print os.path.join(dataset_dir,"closedLeftEyes",img_file)
    for img_file in os.listdir(os.path.join(dataset_dir,"openLeftEyes")):
        if img_file.endswith(".jpg"):
            file_locations += [os.path.join(dataset_dir,"openLeftEyes",img_file)]
            opened +=[1]
        else:
            print os.path.join(dataset_dir,"closedLeftEyes",img_file)
    output_left = pd.DataFrame(columns=["file_location","opened"])
    output_left["file_location"] = file_locations
    output_left["opened"] = opened
    output_left = output_left.sample(frac = 1)

    mask = np.random.rand(len(output_left)) < 0.8
    train = output_left[mask].reset_index(drop=True)
    test = output_left[~mask].reset_index(drop=True)
    train.to_pickle(os.path.join(dataset_dir,"train_left.pkl"))
    test.to_pickle(os.path.join(dataset_dir,"test_left.pkl"))
    print "splitted left eye dataset"
def split_train_test_right_eye_dataset(dataset_dir):
    file_locations = []
    opened = []
    for img_file in os.listdir(os.path.join(dataset_dir,"closedRightEyes")):
        if img_file.endswith(".jpg"):
            file_locations += [os.path.join(dataset_dir,"closedRightEyes",img_file)]
            opened +=[0]
        else:
            print os.path.join(dataset_dir,"closedLeftEyes",img_file)
    for img_file in os.listdir(os.path.join(dataset_dir,"openRightEyes")):
        if img_file.endswith(".jpg"):
            file_locations += [os.path.join(dataset_dir,"openRightEyes",img_file)]
            opened +=[1]
        else:
            print os.path.join(dataset_dir,"closedLeftEyes",img_file)
    output_right = pd.DataFrame(columns=["file_location","opened"])
    output_right["file_location"] = file_locations
    output_right["opened"] = opened
    output_right = output_right.sample(frac = 1)

    mask = np.random.rand(len(output_right)) < 0.8
    train = output_right[mask].reset_index(drop=True)
    test = output_right[~mask].reset_index(drop=True)
    train.to_pickle(os.path.join(dataset_dir,"train_right.pkl"))
    test.to_pickle(os.path.join(dataset_dir,"test_right.pkl"))
    print "splitted right eye dataset"
