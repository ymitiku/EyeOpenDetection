# EyeOpenDetection
Python module which detects if eye is open or closed.



## Dataset used
This project uses [Closed Eyes In The Wild (CEW)](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html) dataset.

## How to split dataset into train and test(pickle files)
The training program expects two pickle files inside `dataset_dir`(option for training program). 
This two files are train.pkl and test.pkl which contains file locactions to training and test sets images respectively. 
The following program can be used to split dataset into train and test sets and save the file locations in respective pickle files.
```
python -m preprocess [options]
```
#### Options to the program
* --dataset_dir - Directroy which contains `ClosedFace` folder and `OpenFace` folder.

## How to run training program
Training program expects dataset dir which contains `ClosedFace` and `OpenFace` folders, and train.pkl and test.pkl files.
The following program can be used to train the model.
```
python -m train [options]
```


## How to run Demo project

## LICENSE

## References
