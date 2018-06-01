# EyeOpenDetection
Training project to detect if eye is open or closed

## How to run EyeOpenDetection
### Train module

### To run multi input model use
```
python -m train --network mi --dataset_dir /path/to/dataset of the model --output /path/to/output file --image_size image_height image_width
```
### To run single input model use
```
python -m train --network si --dataset_dir /path/to/dataset of the model --output /path/to/output file --image_size image_height image_width
```

Where 
* ```--m``` module type. It can be either ```train``` or ```demo```. 
* ```--network``` network type. It can be either ```mi``` or ```si```. 
* ```--output``` path to model's weight output file.
* ```--epochs``` specifies the number of passes through the dataset.
* ```--batch_size``` refers to the number of training examples utilised in one iteration usually between 1 and size of the dataset.
* ```--lr``` the learning rate of the model.
* ```--steps``` how many steps per epoch.
* ```--weights``` path to the weights of the model .h5 file.
* ```--image_size``` two arguments that specify the height and width of the image.


### Dependancies

* tensorflow >= 1.0
* keras >= 2.0
* opencv >= 3.0
* dlib 
* numpy

* [shape_predictor_68_face_landmarks.dat][sp]

#### N.B

* **opencv should be compiled with ffmpeg support.**
* **Conda virtual environment can be created using the following command.**

 ```
 conda env create -f requirements.yml -n emopy_2
 ```

