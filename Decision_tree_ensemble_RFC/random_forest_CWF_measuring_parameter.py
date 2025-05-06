
'''
importing standard python packages
'''
from cProfile import label
from multiprocessing import Value
from pickle import STOP
from pyexpat import model
import sys
import numpy as np
import random
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from tqdm import tqdm


#------------------
'''
Importing the ML based packages
'''
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import shap

#-------------------------------------------------

np.set_printoptions(precision=2)

'''
Opening the polished data table which has preprocessed
'''

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


#checking with whole dataset
df = pd.read_csv("../../final_data_polished_for_all_rails_WW18.csv") # this is the raw dataset where burnt and intact ration is skewed

columns = [cols for cols in df.columns if ("VCCCORE"  in cols or "VCCINF" in cols) and "TPI_SIUP" in cols]
cols_export= ["category"]+columns
df2 = df[cols_export]


features, category = df2.drop('category', axis=1), df2['category']


# Get 80% of the dataset as the training set. Put the remaining 20% in val/testing variables.
X_train, X_val, y_train, y_val = train_test_split(features, category, stratify=category, test_size=0.20, random_state=42)


min_samples_split_list = [2,10, 30, 50, 100]  ## If the number is an integer, then it is the actual quantity of samples,
                                             ## If it is a float, then it is the percentage of the dataset
max_depth_list = [2, 4, 8, 16, 32, 64]
n_estimators_list = [10,50,100,500]




#-- calculating the parameter details -----------------


#===== test 1 ============================
accuracy_list_train = []
accuracy_list_val = []

for min_samples_split in tqdm(min_samples_split_list):
    # you can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(min_samples_split = min_samples_split,
                                   random_state = 42).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## the predicted values for the train dataset
    predictions_val = model.predict(X_val) ## the predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('train x validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['train','validation'])
plt.show()

#===== test 2 ============================

for max_depth in tqdm(max_depth_list):
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = 42).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()

#===== test 3 ============================

for n_estimators in tqdm(n_estimators_list):
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = 42).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
plt.show()