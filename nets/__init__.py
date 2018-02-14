from nets.model import EyeClosedModel
import keras


class EyeClosedNet(object):
    def __init__(self,dataset):
        self.dataset = dataset
        eye_closed_model = EyeClosedModel((24,24,1))
        self.model = eye_closed_model
    def train(self,left_eye):
        
        X_test = self.dataset.test_images
        y_test = self.dataset.test_opened
        if left_eye:
            model = self.model.get_model_with_output_layer("left_eye_open")
        else:
            model = self.model.get_model_with_output_layer("right_eye_open")
        model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(1e-4),metrics=["accuracy"])
        model.fit_generator(self.dataset.generator(32),epochs=20,
            steps_per_epoch=1000,verbose=True,validation_data=[X_test,y_test])
        score = model.evaluate(X_test,y_test)
        if left_eye:
            self.model.model.save_weights("models/left_eye_large.h5")
        else:
            self.model.model.save_weights("models/right_eye_large.h5")
        print "Score",score