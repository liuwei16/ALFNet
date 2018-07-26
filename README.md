# Learning Efficient Single-stage Pedestrian Detectors by Asymptotic Localization Fitting
This is the Keras implementation of our paper accepted in ECCV 2018.
## Introduction
This paper is a step forward pedestrian detection for both speed and accuracy. Specifically, a structurally simple but effective module called Asymptotic Localization Fitting (ALF) is proposed, which stacks a series of predictors to directly evolve the default anchor boxes step by step into improving detection results. As a result, during training the latter predictors enjoy more and better-quality positive samples, meanwhile harder negatives could be mined with increasing IoU thresholds. On top of this, an efficient single-stage pedestrian detection architecture (denoted as ALFNet) is designed, achieving state-of-the-art performance on CityPersons and Caltech. For more details, please refer to our [paper]

 ![img01](./docs/network.png)
