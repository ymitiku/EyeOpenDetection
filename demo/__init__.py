import pandas as pd
import numpy as np 
import os
import cv2
import dlib
from keras.models import model_from_json
import json
from nets.model import EyeClosedModel
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

def get_dlib_points(img,predictor,rectangle):
    shape = predictor(img,rectangle)
    dlib_points = np.zeros((68,2))
    for i,part in enumerate(shape.parts()):
        dlib_points[i] = [part.x,part.y]
    return dlib_points
def get_left_eye(face_img,dlib_points):
    left_eye_left = int(max(dlib_points[36][0]-5,0))
    left_eye_top = int(max(min(dlib_points[37][1],dlib_points[38][1])-5,0))
    left_eye_right = int(min(face_img.shape[1],dlib_points[39][0]+5))
    left_eye_bottom = int(min(max(dlib_points[40][1],dlib_points[41][1])+5,face_img.shape[0]))
    left_eye = face_img[left_eye_top:left_eye_bottom,left_eye_left:left_eye_right]

    return left_eye
def get_right_eye(face_img,dlib_points):
    right_eye_left = int(max(dlib_points[42][0]-5,0))
    right_eye_top = int(max(min(dlib_points[43][1],dlib_points[45][1])-5,0))
    right_eye_right = int(min(face_img.shape[1],dlib_points[45][0]+5))
    right_eye_bottom = int(min(max(dlib_points[46][1],dlib_points[47][1])+5,face_img.shape[0]))
    right_eye = face_img[right_eye_top:right_eye_bottom,right_eye_left:right_eye_right]

    return right_eye

def eye_open_webcam_demo(model_path):
    eyeclosed_model = EyeClosedModel((24,24,1))
    model = eyeclosed_model.model
    model.load_weights(model_path)
    cap = cv2.VideoCapture(-1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # model.summary()
    while cap.isOpened():
        _,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for i,face in enumerate(faces):
            face_img = gray[
                    max(0,face.top()):min(gray.shape[0],face.bottom()),
                    max(0,face.left()):min(gray.shape[1],face.right())
            ]
            cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),color=(255,0,0),thickness=2)
            face_img = cv2.resize(face_img,(100,100))
            dlib_points = get_dlib_points(face_img,predictor,dlib.rectangle(0,0,face_img.shape[1],face_img.shape[0]))
            left_eye  = get_left_eye(face_img,dlib_points)
            left_eye = cv2.resize(left_eye,(24,24))
            right_eye = get_right_eye(face_img,dlib_points)
            right_eye = cv2.resize(right_eye,(24,24))
            left_eye = left_eye.astype(np.float32)/255
            right_eye = right_eye.astype(np.float32)/255
            left_prediction = model.predict(left_eye.reshape(-1,24,24,1))[1]
            right_prediction = model.predict(right_eye.reshape(-1,24,24,1))[0]
            left_arg_max = np.argmax(left_prediction)
            right_arg_max = np.argmax(right_prediction)
            if left_arg_max ==0:
                left_text = "Left eye Closed"
            else:
                left_text = "Left eye Opened"
            if right_arg_max ==0:
                right_text = "Right eye Closed"
            else:
                right_text = "Right eye Opened"

            cv2.putText(frame,left_text,(face.left()+10,face.top()+10), cv2.FONT_HERSHEY_DUPLEX, 0.4,color=(0,0,255),thickness=1)
            cv2.putText(frame,right_text,(face.left()+10,face.top()+30), cv2.FONT_HERSHEY_DUPLEX, 0.4,color=(0,0,255),thickness=1)
            cv2.imshow("Frame",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()
            
        
