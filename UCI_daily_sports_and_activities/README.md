# sports_activities_dataset
Daily sports activity dataset from UCI ML repository.

Dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)

The **important** ipynb is [csir_cdri_test.ipynb](https://github.com/greed2411/sports_activities_dataset/blob/master/csir_cdri_test.ipynb), secondary file is [prediction-pca.ipynb](https://github.com/greed2411/sports_activities_dataset/blob/master/prediction-pca.ipynb).

**POST SUBMISSION edit** : Pytorch results are in [pytorch-model.ipynb](https://github.com/greed2411/sports_activities_dataset/blob/master/pytorch%20model.ipynb). So look into it as well.

# Report: 

(also part of csir_cdri_test.ipynb)

It has been done according to the **research** paper,

K. Altun, B. Barshan, and O. Tunçel,
`Comparative study on classifying human activities with miniature inertial and magnetic sensors`,
Pattern Recognition, 43(10):3605-3620, October 2010.

# Preprocessing

Where they take the input segment, that's a 5 second window of a patient performing an activity, which has 125 observations ( 5 x 25Hz ) with 45 features, because of 9 axes of each sensor unit on torso, left hand, right hand, left leg, right leg. They convert the 125x45 into a handcrafted meaningful 1170x1 matrix.

The 1170 features represents, 
* 225 features ( min, max, mean, skewness, kurtosis of all 9 axes of all 5 units, thus 5x9x5 ) i.e., first_step
* 225 features which represent the maximum 5 peaks of the DFT applied on each of the 9 axes of all the 5 units i.e.,
    second_step
* 225 features which represent the corresponding frequency of the 5 peaks of the DFT over the time series i.e.,
    third_step.
* 495 features which represent the autocorrelation of the series, 11 hand picked values from the 125 
    autocorrelation values for each axes, thus 11 x 9 x 5 = 495 i.e, fourth_step.
    
Adding them all 225 + 225 + 225 + 495 = 1170, for each segment, i.e., each text file.

Then these values are normalized in the range [0,1], and stored along with the patient ID and activity ID for that segment / text file.

The test I performed includes two parts, one with 9120 x 1170 matrix, and another with 9120 x 30 matrix, I did PCA over the initial matrix, unfortunately, the PCA didn't live up to my expectations, so I'm producing both results.

# Conclusion

## Actual Dataset, 9120 x 1172

### Predicting Activity with signals, on actual dataset.


|**Model**                    |**Accuracy**|
|-----------------------------|------------|
|Gradient Boosting Classifier | 0.9368 |
|Bagging Classifier|0.9100|
|Random Forest Classifier|0.9017|
|ExtraTrees Classifier|0.8872|
|Decision Tree|0.8552|

*Runner up*: Neural Networks (DNN) : 0.85

### Predicting Patient with signals and activity, The bonus task, on actual dataset


|**Model**                    |**Accuracy**|
|-----------------------------|------------|
|Bagging Classifier|0.8245|
|Gradient Boosting Classifier | 0.7921 |
|Random Forest Classifier|0.7627|
|Decision Tree|0.7394|
|kNN (k=3) |0.6578|


## PCA Dataset, 9120 x 32, available as a .csv file in the repo.

### Predicting Activity with signals, on PCA dataset


|**Model**                    |**Accuracy**|
|-----------------------------|------------|
|ExtraTrees Classifier|0.8767|
|Gradient Boosting Classifier | 0.8745 |
|Random Forest Classifier|0.8596|


*Runner up*: Bagging Classifier : 0.8320, Neural Networks (DNN) : 0.8192

#### With Neural Networks (PyTorch) on PCA dataset

|**Model**                    |**Accuracy**|
|-----------------------------|------------|
|With Adam Optimizer  |0.8951|
|With Adam Optimizer, Karpathy constant | 0.8078 |
|With RMSProp optimizer  |0.8877|
|With SGD Optimizer | 0.8451 |


#### With Neural Networks (scikit-learn) on PCA dataset


|**Model**                    |**Accuracy**|
|-----------------------------|------------|
|MLP with Adam + ReLU |0.8065|
|MLP with Adam + Sigmoid | 0.7771 |


MLP with SGD are worse than I expected, they converged within 1000 epochs, with accuracy < 40%, whereas Adam optmizers made it possible with less than 500 epochs and accuracy > 75%.


### Bonus task with PyTorch on PCA dataset.


|**Model**                    |**Accuracy**|
|-----------------------------|------------|
|With RMSProp optimizer  |0.5043|

For NNs, `Pytorch` examples are considered far well from `scikit-Learn` whereas, hyperparameter tuning is the most important, I did my best to produce the above results.


***END***


EDIT Jan 16 2018 15:38 hrs: Postsubmission edits are about Pytorch results.

