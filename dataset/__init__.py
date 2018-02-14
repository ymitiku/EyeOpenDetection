import numpy as np 
import pandas as pd 
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator


class EyeClosedDataset(object):
    def __init__(self,dataset_dir,image_shape=(24,24),left_eye=True):
        if not os.path.exists(dataset_dir):
            print("Dataset path ",dataset_dir," does not exist")
            exit(0)
        self.dataset_dir = dataset_dir
        self.image_shape = image_shape
        self.left_eye = left_eye
        self.dataset_loaded = False
    def load_dataset(self):
        eye = "right"
        if self.left_eye:
            eye = "left"
       
        train = pd.read_pickle(os.path.join(self.dataset_dir,"train_"+eye+".pkl")).reset_index(drop=True)
        test = pd.read_pickle(os.path.join(self.dataset_dir,"test_"+eye+".pkl")).reset_index(drop=True)
        train_opened = train["opened"].as_matrix()
        test_opened = test["opened"].as_matrix()

        self.train_images = self.load_images(train).astype(np.float32)/255
        self.test_images = self.load_images(test).astype(np.float32)/255

        self.train_images = self.train_images.reshape(-1,self.image_shape[0],self.image_shape[1],1)
        self.test_images = self.test_images.reshape(-1,self.image_shape[0],self.image_shape[1],1)
        self.train_opened = np.eye(2)[train_opened]
        self.test_opened = np.eye(2)[test_opened]
        self.dataset_loaded = True
    def load_images(self,dataframe):
        output = np.zeros((len(dataframe),self.image_shape[0],self.image_shape[1]))
        for index,row in dataframe.iterrows():
            img = cv2.imread(row["file_location"])
            if img is None:
                print "Cv2 error: Unable to read from '"+row["file_location"]+"'"
                continue
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            output[index] = img 
        return output
    def generator(self,batch_size):
        if not self.dataset_loaded:
            print "Dataset is not loaded"
            exit(0)
        datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
                
                )
        
        return datagen.flow(self.train_images,self.train_opened,batch_size=batch_size)
    