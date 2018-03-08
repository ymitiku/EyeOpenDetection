from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Model
import numpy as np
import keras

class MultiInputEyeStateModel(object):
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.model = self.build()
    def build(self):
        image_input_layer = Input(shape=self.input_shape)
        image_layer = Conv2D(32,kernel_size=3,strides=1,padding='same',activation="relu")(image_input_layer)
        image_layer = MaxPooling2D(pool_size=(2,2))(image_layer)
        image_layer = Conv2D(64,kernel_size=3,strides=1,padding='same',activation="relu")(image_layer)
        image_layer = MaxPooling2D(pool_size=(2, 2))(image_layer)
        image_layer = Flatten()(image_layer)
        # order of key points for left eye
        #    18,19,20,21,22,37,38,39,40,41,42
        # order of key points for right eye
        #    23,24,25,26,27,43,44,45,46,47,48
        key_points_input_layer = Input(shape=(1,6,2))
        key_points = Conv2D(8,kernel_size=(1,3),strides=(1, 1),padding='same',activation="relu",kernel_initializer='glorot_uniform')(key_points_input_layer)
        key_points = Conv2D(16,kernel_size=(1,3),strides=(1, 1),padding='same',activation="relu",kernel_initializer='glorot_uniform')(key_points)
        key_points = Flatten()(key_points)
    
        dists_input_layer = Input(shape=(1,6,1))
        dists = Conv2D(8,kernel_size=(1,3),strides=(1, 1),padding='same',activation="relu",kernel_initializer='glorot_uniform')(dists_input_layer)
        dists = Conv2D(16,kernel_size=(1,3),strides=(1, 1),padding='same',activation="relu",kernel_initializer='glorot_uniform')(dists)
        dists = Flatten()(dists)

        angles_input_layer = Input(shape=(1,6,1))
        angles = Conv2D(8,kernel_size=(1,3),strides=(1, 1),padding='same',activation="relu",kernel_initializer='glorot_uniform')(angles_input_layer)
        angles = Conv2D(16,kernel_size=(1,3),strides=(1, 1),padding='same',activation="relu",kernel_initializer='glorot_uniform')(angles)
        angles = Flatten()(angles)
        
        aspect_ratio_input_layer = Input(shape=(1,))
        aspect_ratio_layer = Dense(8)(aspect_ratio_input_layer)
        aspect_ratio_layer = Dense(16)(aspect_ratio_layer)


        merged_layers = keras.layers.concatenate([image_layer, key_points,dists,angles,aspect_ratio_layer])
        
        merged_layers = Dense(128, activation='relu')(merged_layers)
        merged_layers = Dropout(0.2)(merged_layers)
        merged_layers = Dense(256, activation='relu')(merged_layers)
        merged_layers = Dropout(0.2)(merged_layers)
        merged_layers = Dense(2, activation='softmax')(merged_layers)
        model = Model(inputs=[image_input_layer, key_points_input_layer,dists_input_layer,angles_input_layer,aspect_ratio_input_layer],outputs=merged_layers)
        return model


class MultiInputEyeStateNet(object):
    def __init__(self,dataset,epochs = 10,batch_size=32,lr = 1e-4,steps_per_epoch=200,weights=None,output=None):
        self.dataset = dataset
        eye_closed_model = MultiInputEyeStateModel((24,24,1))
        self.model = eye_closed_model

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.weights = weights
        self.output = output
        if not self.weights is None:
            print "loading weights from ",self.weights
            self.model.model.load_weights(weights)
            print "loaded model weights"
    def train(self):
        
        if not self.dataset.dataset_loaded:
            self.dataset.load_dataset()
        eye_images = self.dataset.test_images
        key_points = self.dataset.test_key_points
        dists = self.dataset.test_dists
        angles = self.dataset.test_angles
        aspect_ratios = self.dataset.test_aspect_ratios
        key_points = np.expand_dims(key_points,1)
        dists = np.expand_dims(dists,1)
        angles = np.expand_dims(angles,1)

        eye_images = eye_images.astype(np.float32)/255
        key_points = key_points.astype(np.float32)/24
        dists = dists.astype(np.float32)/24
        angles = angles.astype(np.float32)/np.pi
     
        X_test = [eye_images,key_points,dists,angles,aspect_ratios]
        y_test = self.dataset.test_opened

        y_test = np.eye(2)[y_test.astype(np.uint8)]

      
        model = self.model.model
        model.summary()
        model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(self.lr),metrics=["accuracy"])
        model.fit_generator(self.dataset.generator(self.batch_size),epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,verbose=True,validation_data=[X_test,y_test])
        score = model.evaluate(X_test,y_test)
        self.model.model.save_weights("models/"+self.output+".h5")
        with open("logs/log.txt","a+") as log_file:
            log_file.write(self.output+" score: "+str(score))
        print "Score",score
