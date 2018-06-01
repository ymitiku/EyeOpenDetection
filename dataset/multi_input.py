import os 
import cv2
import dlib 
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

class MultiInputEyeStateDataset(object):
    def __init__(self,dataset_dir,predictor,image_shape=(24,24)):
        self.dataset_dir = dataset_dir
        self.image_shape = image_shape
        self.predictor = predictor
        self.dataset_loaded = False
    
    def get_attributes_from_local_frame(self,face_image,key_points_11):
        
        face_image_shape = face_image.shape
        top_left = key_points_11.min(axis=0)
        bottom_right = key_points_11.max(axis=0)

        # bound the coordinate system inside eye image
        bottom_right[0] = min(face_image_shape[1],bottom_right[0])
        bottom_right[1] = min(face_image_shape[0],bottom_right[1]+5)
        top_left[0] = max(0,top_left[0])
        top_left[1] = max(0,top_left[1])

        # crop the eye
        top_left = top_left.astype(np.uint8)
        bottom_right = bottom_right.astype(np.uint8)
        eye_image = face_image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
        # translate the eye key points from face image frame to eye image frame
        key_points_11 = key_points_11 - top_left
        key_points_11 +=np.finfo(float).eps
        # horizontal scale to resize image
        scale_h = self.image_shape[1]/float(eye_image.shape[1])
        # vertical scale to resize image
        scale_v = self.image_shape[0]/float(eye_image.shape[0])

        # resize left eye image to network input size
        eye_image = cv2.resize(eye_image,(self.image_shape[0],self.image_shape[1]))

        # scale left key points proportional with respect to left eye image resize scale
        scale = np.array([[scale_h,scale_v]])
        key_points_11 = key_points_11 * scale 

        # calculate centroid of left eye key points 
        centroid = np.array([key_points_11.mean(axis=0)])

        # calculate distances from  centroid to each left eye key points
        dists = self.distance_between(key_points_11,centroid)

        # calculate angles between centroid point vector and left eye key points vectors
        angles = self.angles_between(key_points_11,centroid)
        return eye_image, key_points_11,dists,angles

    def get_left_eye_attributes(self,face_image):
        
        face_image_shape = face_image.shape
        face_rect = dlib.rectangle(0,0,face_image_shape[1],face_image_shape[0])
        kps = self.get_dlib_points(face_image,self.predictor,face_rect)
        # Get key points of the eye and eyebrow

        key_points_11 = self.get_left_key_points(kps)
        
        eye_image,key_points_11,dists,angles = self.get_attributes_from_local_frame(face_image,key_points_11)
        face_image = self.draw_key_points(face_image,kps)

        
        return eye_image,key_points_11,dists,angles

    def draw_key_points(self,image,kps):
        for kp in kps:
           image = cv2.circle(image,(int(kp[0]),int(kp[1])),1,(0,255,0))
        return image

    def get_right_eye_attributes(self,face_image):
        
        face_image_shape = face_image.shape
        face_rect = dlib.rectangle(0,0,face_image_shape[1],face_image_shape[0])
        kps = self.get_dlib_points(face_image,self.predictor,face_rect)
        # Get key points of the eye and eyebrow

        key_points_11 = self.get_right_key_points(kps)
        
        eye_image,key_points_11,dists,angles = self.get_attributes_from_local_frame(face_image,key_points_11)

        return eye_image,key_points_11,dists,angles
        
    def distance_between(self,v1,v2):
        diff = v2 - v1
        diff_squared = np.square(diff)
        dist_squared = diff_squared.sum(axis=1) 
        dists = np.sqrt(dist_squared)
        return dists

    def angles_between(self,v1,v2):
        dot_prod = (v1 * v2).sum(axis=1)
        v1_norm = np.linalg.norm(v1,axis=1)
        v2_norm = np.linalg.norm(v2,axis=1)
        

        cosine_of_angle = (dot_prod/(v1_norm * v2_norm)).reshape(11,1)

        angles = np.arccos(np.clip(cosine_of_angle,-1,1))
        return angles

    def get_right_key_points(self,key_points):
        output = np.zeros((11,2))
        output[0:5] = key_points[17:22]
        output[5:11] = key_points[36:42]
        return output

    def get_left_key_points(self,key_points):
        output = np.zeros((11,2))
        output[0:5] = key_points[22:27]
        output[5:11] = key_points[42:48]
        return output

    def load_dataset(self):
        if not os.path.exists(os.path.join(self.dataset_dir,"train.pkl")):
            print("Required file train.pkl does not exist inside ",self.dataset_dir," please split dataset into train and test before using this class instance.")
            exit(0)
        if not os.path.exists(os.path.join(self.dataset_dir,"test.pkl")):
            print("Required file test.pkl does not exist inside ",self.dataset_dir," please split dataset into train and test before using this class instance.")
            exit(0)
        train_df = pd.read_pickle(os.path.join(self.dataset_dir,"train.pkl")).reset_index(drop=True)
        test_df = pd.read_pickle(os.path.join(self.dataset_dir,"test.pkl")).reset_index(drop=True)
        self.train_opened = train_df["opened"].as_matrix().astype(np.uint8)
        self.test_opened = test_df["opened"].as_matrix().astype(np.uint8)
    
        self.train_face_images = self.load_face_images(train_df)
        test_face_images = self.load_face_images(test_df)
        
        self.test_images,self.test_key_points,self.test_dists,self.test_angles,self.test_opened = self.__load_dataset(test_face_images,self.test_opened)

        self.dataset_loaded = True

    def load_face_images(self,df):
        output_images = np.zeros((len(df),100,100))
        for index,row in df.iterrows():
            img = cv2.imread(row["file_location"].replace("mtk", "samuel"))
            if img is None:
                print ("cv2 error: Unable to read from ",row["file_location"])
                continue
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            output_images[index] = img
        return output_images

    def __load_dataset(self,imgs,opened):
        output_eye_images = np.zeros((len(imgs)*2,self.image_shape[0],self.image_shape[1],1))
        output_eye_key_points = np.zeros((len(imgs)*2,11,2))
        output_eye_kp_dists = np.zeros((len(imgs)*2,11,1))
        output_eye_kp_angles = np.zeros((len(imgs)*2,11,1))
        output_opened = np.zeros((len(opened)*2))
        for index in range(len(imgs)):
            img = imgs[index].astype(np.uint8)
            img = img.reshape(100,100)
            for j in range(2):
                if j==0:
                    eye_image,kp,kp_dists,kp_angles = self.get_left_eye_attributes(img)
                else:
                    eye_image,kp,kp_dists,kp_angles = self.get_right_eye_attributes(img)
                output_eye_images[index*2+j] = eye_image.reshape(-1,self.image_shape[0],self.image_shape[1],1)
                output_eye_key_points[index*2+j] = kp.reshape(11,2)
                output_eye_kp_dists [index*2+j] = kp_dists.reshape(11,1)
                output_eye_kp_angles[index*2+j] = kp_angles.reshape(11,1)
                output_opened[index*2+j] = opened[index]
                
           
        return output_eye_images,output_eye_key_points,output_eye_kp_dists,output_eye_kp_angles,output_opened

    def get_dlib_points(self,img,predictor,rectangle):
        shape = predictor(img,rectangle)
        dlib_points = np.zeros((68,2))
        for i,part in enumerate(shape.parts()):
            dlib_points[i] = [part.x,part.y]
        return dlib_points

    def generator(self,batch_size):
        datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                shear_range=0.1,
                horizontal_flip=True
                
                )
        self.train_face_images = self.train_face_images.reshape(-1,100,100,1)
        current_generator  = datagen.flow(self.train_face_images,self.train_opened,batch_size)
        while True:
            
            face_images,opened = current_generator.next()
            eye_images,key_points,dists,angles,new_opened = self.__load_dataset(face_images,opened)
            key_points = np.expand_dims(key_points,1)
            dists = np.expand_dims(dists,1)
            angles = np.expand_dims(angles,1)

            eye_images = eye_images.astype(np.float32)/255
            key_points = key_points.astype(np.float32)/24
            dists = dists.astype(np.float32)/24
            angles = angles.astype(np.float32)/np.pi

            
            X = [eye_images,key_points,dists,angles]
            y = np.eye(2)[new_opened.astype(np.uint8)]
            yield X,y


            # indexes = np.arange(len(self.train_images))
            # np.random.shuffle(indexes)
            # for i in range(0,len(indexes)-batch_size,batch_size):
            #     current_indexes = indexes[i:i+batch_size]
            #     images = self.train_images[current_indexes]
            #     key_points = self.train_key_points[current_indexes]
            #     dists = self.train_dists[current_indexes]
            #     angles = self.train_angles[current_indexes]

            #     key_points = np.expand_dims(key_points,1)
            #     dists = np.expand_dims(dists,1)
            #     angles = np.expand_dims(angles,1)

            #     X = [images,key_points,dists,angles]
            #     y = self.train_opened[current_indexes]
            #     y = np.eye(2)[y]
            #     yield X,y
