# Lung Nodule Type Classification
The main objective of this project is to develop type classification model for lung nodule in 3D CT image.
There are 3 classes of solid, part-solid and non-solid in lung nodule type classification.

<img src="assets/img/type_classification_01.jpg" width="40%" height="30%" title="Lung Nodule Type" alt="lung_nodule_type"></img>

# Table Of Contents
-  [Requirements](#Requirements)
-  [How to use](#how-to-use)
-  [In Details](#in-details)

# Requirements
- [Python](https://www.python.org/) Ensure Python 3.0 or greater is installed.
- [Tensorflow](https://www.tensorflow.org) An open source deep learning platform. v 1.15.5

# How to use
## **1. Preprocessing**
- Before training, cropped data have to be prepared
- The cropped data is formed as **numpy array**
- The shape of array is very important for training model. Set the crop size considering target object. In this project, the crop size is 48x 32x48

## **2. Train a model**
- Run `train.py` with correct arguments
```bash
python train.py --num_gpu 2 --batch_size 64 --lr_init 0.001 --save_path ./weights --data_path ./data --CUDA_VISIBLE_DEVICES 0,1
```

## **3. Validation a trained model**
- Run `validation.py` with correct arguments
- The weights from training step should be saved in `save_path`
```bash
python validation.py --num_gpu 2 --batch_size 64 --lr_init 0.001 --save_path ./weights --data_path ./data --CUDA_VISIBLE_DEVICES 0,1
```

## **4. Analysis of a trained model**
- Run `analysis.py` with correct arguments
- The weights from training step should be saved in `save_path`
```bash
python analysis.py --num_gpu 2 --batch_size 64 --lr_init 0.001 --save_path ./weights --data_path ./data --CUDA_VISIBLE_DEVICES 0,1
```

## **5. Test a trained model**
- Run `test.py` with correcnt arguments
- The weights from training step should be saved in `save_path`
```bash
python test.py --num_gpu 2 --batch_size 64 --lr_init 0.001 --save_path ./weights --data_path ./data --CUDA_VISIBLE_DEVICES 0,1
```

# In Details
```
├──  analysis.py - here's the file to analyze the results
│
├──  model.py - here's the file of classification model
│
├──  test.py - this file contains the inference process
│
├──  model.py - here's the file to train classification model
│
├──  utils.py - here's the file including utilities for training and inference
│
└──  validation.py - this file contains the evaludation process with        valdiation dataset
```



