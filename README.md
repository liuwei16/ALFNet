# Progressive Refinement Network for Occluded Pedestrian Detection
Keras implementation of PRNet accepted in ECCV 2020. 

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







