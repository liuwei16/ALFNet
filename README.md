# Progressive Refinement Network for Occluded Pedestrian Detection
Keras implementation of PRNet accepted in ECCV 2020. 

## Introduction
This paper presents Progressive Refinement Network (PRNet), a novel single-stage detector that tackles occluded pedestrian detection. Motivated by human¡¯s progressive process on annotating occluded pedestrians, PRNet achieves sequential refinement by three phases: Finding high confident anchors of visible parts, calibrating such anchors to a full-body template based on occlusion statistics, and then predicting final full-body regions from the calibrated anchors. For more details, please refer to our paper.

### Dependencies

* python 2.7
* numpy 1.12.0
* Tensorflow 1.x
* keras 2.0.6
* OpenCV

### Get Start
1. Get the code.
```
git clone https://github.com/sxlpris/PRNet.git
```
2. Install the requirements.
```
  pip install -r requirements.txt
``` 
3. Download the dataset [CityPersons](https://bitbucket.org/shanshanzhang/citypersons) to folder '$PRNet/data/cityperson/'.

2. Download the initialized models [ResNet-50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5) to folder '$PRNet/data/models/'.

4. Train.
```
run *train_prnet.py*
```
5. Test.
```
run *test_prnet.py*
```

### Model
To help reproduce the results in our paper, we provide our model [PRNet_city.hdf5](https://pan.baidu.com/s/1VKs9YogMPBjiQdKm6MAuLA) (password:imiq) trained on CityPersons.







