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

## Get Started
In this project, we offer you several options to explore our methods:
1. **Option1:** Use the pre-computed soft-labels. You can ideally reproduced the results on kaggle
2. **Option2:** Use the pre-trained model. You can re-run all the inference and get soft labels from different model. Then, you can ensemble them to get the final result (The results might be slightly different from Kaggle leaderboard, but should be similar.)
3. **Option3:** Use the provided training scripts and external data. You can step-by-step reproduce all the process from training on external => training on cil dataset => ensemble.

### Option1
Run the provided ensemble code by
```bash
cd $PROJECT_ROOT
python ensemble_soft_labels.py
``` 
```submission_0.csv``` will be created at ```$PROJECT_ROOT```

### Option2
Run the provided testing scripts in ```$PROJECT_ROOT\scripts```<br/>
First go to ```$PROJECT_ROOT```
```bash
cd $PROJECT_ROOT
```
1. Standard cross-entropy model:
   ```bash
   sh ./scripts/test_deeplabv3+_class_export.sh
   ```
2. Cross-entropy model trained in old data sampling method:
   ```bash
   sh ./scripts/test_deeplabv3+_class_old_export.sh
   ```
3. Cross-entropy model trained using only 90 images from cil dataset:
   ```bash
   sh ./scripts/test_deeplabv3+_class_90_export.sh
   ```
4. Focal loss model:
   ```bash
   sh ./scripts/test_deeplabv3+_focal_export.sh
   ```
5. Weighted cross-entropy model:
   ```bash
   sh ./scripts/test_deeplabv3+_wclass_export.sh
   ```
6. Weighted cross-entropy model with larger learning rate:
   ```bash
   sh ./scripts/test_deeplabv3+_wclass_new_export.sh
   ```
7. Finally, if all the previous steps are done correctly, we can ensemble them and produce the final results:
   ```bash
   python ensemble_soft_labels.py
   ```