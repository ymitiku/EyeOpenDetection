import cv2
import dlib
import numpy as np
from demo import split_train_test_left_eye_dataset,split_train_test_right_eye_dataset
import argparse
def get_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir",default="",type=str)
    parser.add_argument("--left",default=True,type=bool)

    args = parser.parse_args()
    return args

def main():
    # img = cv2.imread("imgs/image00041.jpg")
    # height= img.shape[0]
    # width = img.shape[1]
    # h = 720
    # scale = h/float(height)
    # w = scale * width 
    # img = cv2.resize(img,(int(w),int(h)))

    # detector = dlib.get_frontal_face_detector()
    # faces = detector(img)
    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # face = faces[0]
    # cv2.rectangle(img,(face.left(),face.top()),(face.right(),face.bottom()),color=(255,0,0),thickness=2)
    # dlib_points = np.zeros((68,2))
    # f_top = face.top()
    # f_left = face.left()
    # f_right = face.right()
    # f_bottom = face.bottom()

    # face_img = img[f_top:f_bottom,f_left:f_right]
    # face_img = cv2.resize(face_img,(100,100))
    # shape = predictor(face_img,dlib.rectangle(0,0,face_img.shape[1],face_img.shape[0]))



    # for i,part in enumerate(shape.parts()):
    #     # if i in range(36,42):
    #         # cv2.circle(face_img,(part.x,part.y),2,color=(0,255,0),thickness=2)
    #     if i in range(42,48):
    #         cv2.circle(face_img,(part.x,part.y),2,color=(0,0,255),thickness=2)
    #     dlib_points[i] = [part.x,part.y]
    
    # left_eye_left = int(max(dlib_points[36][0]-5,0))
    # left_eye_top = int(max(min(dlib_points[37][1],dlib_points[38][1])-5,0))
    # left_eye_right = int(min(face_img.shape[1],dlib_points[39][0]+5))
    # left_eye_bottom = int(min(max(dlib_points[40][1],dlib_points[41][1])+5,face_img.shape[0]))

    # left_eye = face_img[left_eye_top:left_eye_bottom,left_eye_left:left_eye_right]
    # left_eye = cv2.resize(left_eye,(24,24))
    # cv2.imshow("left_eye",left_eye)
    # cv2.imshow("face",face_img)
    # cv2.imshow("Image",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    args = get_cmd_args()
    print args.left
    if args.left:
        split_train_test_left_eye_dataset(args.dataset_dir)
    else:
        split_train_test_right_eye_dataset(args.dataset_dir)

if __name__ == "__main__":
    main()