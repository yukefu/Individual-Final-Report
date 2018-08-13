# Individual-Final-Report

#### Datasets
Locate in Datasets folder with defintion under folder directory.

- train.csv : original training dataset. Source file: https://www.kaggle.com/c/shelter-animal-outcomes

- TrainK.csv : Labelencoding datasets before one hot encoding include both Target and outcome variables.

- Train_onehot.csv: one-hot-encoding datasets.

- Train_onehot_target.csv and Train_onehot_outcome.csv : datasets after one-hot-encoding with different target outcomes.those two datasets just are used in decision tree part. 

     - Train_onehot_target.csv and Train_onehot_outcome.csv are created for one-hot-encoding with different response variables (Target, outcome).


#### Code

##### Preprocessing
- label encoding part

- missing data part
     
- one hot encoding & merging data part

##### Decision Tree

- Target part
     - Grid Search
     - fit entropy Decision Tree
     - calculate Classification Report 
     - calculate confusion matrix
     - display decision tree
     - display important features

- outcome part
     - Grid Search
     - fit entropy Decision Tree
     - calculate Classification Report 
     - calculate confusion matrix
     - display decision tree
     - display important features
