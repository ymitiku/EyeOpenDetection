from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Model
class EyeStateModel(object):
    def __init__(self,input_shape):
        self.model = self.build(input_shape)
    def build(self,input_shape):
        input_layer = Input(shape=input_shape)
        conv1 = Conv2D(32,kernel_size=3,strides=1,padding='same')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
        conv2 = Conv2D(64,kernel_size=3,strides=1,padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten1 = Flatten()(pool2)
        dense1 = Dense(256)(flatten1)
        drop1 = Dropout(0.2)(dense1)
        right_eye_open = Dense(2,activation="softmax",name="right_eye_open")(drop1)
        dense2 = Dense(256)(flatten1)
        drop2 = Dropout(0.2)(dense2)
        left_eye_open = Dense(2,activation="softmax",name="left_eye_open")(drop2)
        model = Model(input_layer,outputs=[right_eye_open,left_eye_open])
        return model
    def get_model_with_output_layer(self,layer_name):
        input_layer = self.model.input
        for i in range(len(self.model.layers)):
            if self.model.layers[i].name==layer_name:
                output_layer_output = self.model.layers[i].output 
                model = Model(input_layer,outputs=[output_layer_output])
                return model
        raise Exception("The model doesnot contain layer with name: "+str(layer_name))
