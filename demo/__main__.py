import cv2
import dlib
import numpy as np
from demo import eye_open_webcam_demo,split_train_test_left_eye_dataset,split_train_test_right_eye_dataset
import argparse
from demo import view_face_dataset,show_performance_of_model_on_eye_dataset
def get_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir",default="",type=str)
    parser.add_argument("--left",type=bool)
    parser.add_argument("--process",default="webcam_demo",type=str)

    args = parser.parse_args()
    return args

def main():
   
    args = get_cmd_args()
    if args.process == "webcam_demo":
        eye_open_webcam_demo("models/left2.h5")
    elif args.process == "split dataset":
        if args.left:
            split_train_test_left_eye_dataset(args.dataset_dir)
        else:
            split_train_test_right_eye_dataset(args.dataset_dir)
    elif args.process == "view_dataset":
        view_face_dataset("/home/mtk/dataset/eye-closed/dataset_B_FacialImages/OpenFace")
    elif args.process == "performance":
        show_performance_of_model_on_eye_dataset("models/right_left.h5","/home/mtk/dataset/eye-closed/dataset_B_FacialImages",False)

if __name__ == "__main__":
    main()