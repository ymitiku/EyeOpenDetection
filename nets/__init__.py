from nets.model import EyeClosedModel
import keras


class EyeClosedNet(object):
    def __init__(self,dataset,left_eye=True,epochs = 10,batch_size=32,lr = 1e-4,steps_per_epoch=200,
            weights=None,output=None):
        self.dataset = dataset
        eye_closed_model = EyeClosedModel((24,24,1))
        self.model = eye_closed_model
        self.left_eye = left_eye
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.weights = weights
        self.output = output
    def train(self):
        
        X_test = self.dataset.test_images
        y_test = self.dataset.test_opened
        if self.left_eye:
            model = self.model.get_model_with_output_layer("left_eye_open")
        else:
            model = self.model.get_model_with_output_layer("right_eye_open")
        model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(self.lr),metrics=["accuracy"])
        model.fit_generator(self.dataset.generator(self.batch_size),epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,verbose=True,validation_data=[X_test,y_test])
        score = model.evaluate(X_test,y_test)
        if left_eye:
            self.model.model.save_weights("models/"+self.output+".h5")
        else:
            self.model.model.save_weights("models/"+self.output+".h5")
        print "Score",score