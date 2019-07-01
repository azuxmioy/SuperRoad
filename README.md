# Project repository for Computational Intelligence Lab (2019)

Members (in alphabetical order of last name):
- Davin Choo
- Hsuan-I Ho
- Juan-ting Lin
- Vaclav Rozhon

## Project 3: Road segmentation

CIL Website: http://da.inf.ethz.ch/teaching/2019/CIL/project.php

Kaggle: https://inclass.kaggle.com/c/cil-road-segmentation-2019

## Sample Usage

Look at `example_method.py` and `example_main.py`.

## Some Documentations
* `config/`: Place to put config files for all the methods added after deeplab. 
* `dataset/`: dataset images, dataloader class, etc.
  * `cil_dataloader.py`: tf dataset for cil dataset.
  * `external_dataloader.py`: tf dataset for external dataset.
  * Later we might have second external dataset from CVPR challenge.
* `deeplab/`: codes for deeplab method.
* `methods/`: other baseline methods.
* `nets`: all the deep learning feature extractors.
* `postprocessing`: all the post-processing pipeline related codes.
* `resnet_v1_50`: pre-trained model for resnet50.
* `xception`: pre-trained model for xception.
* `utils`: Put some useful but ugly codes that you don't want to have in main files.

## Brief description of files and classes

`dataset/train_input/`: All training images

`dataset/train_output/`: All training labels

`dataset/test_input/`: All testing images

`util.py`: Put imports and utility functions here.

`dataset.py`: Dataset loader. Add any preprocessing methods here.

`csv2img.py`: Helps to visualize output by reading CSV created

`method.py`: Abstract class for methods. All methods should inherit from abstract class `method.py`.

## Some ToDo
* Fix focal loss
* Add random scaling and color jitter to training
* add smoothness loss
* Testing different combinations
* add crf refinement
* multi-scale aggregation