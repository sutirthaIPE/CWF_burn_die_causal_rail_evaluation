
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
df = pd.read_csv("../../final_data_polished_for_all_rails.csv") # this is the raw dataset where burnt and intact ration is skewed

columns = [cols for cols in df.columns if ("VCCCORE"  in cols or "VCCINF" in cols) and "TPI_SIUP" in cols]
cols_export= ["category"]+columns
df2 = df[cols_export]



#sys.exit()

features, category = df2.drop('category', axis=1), df2['category']

#sys.exit()

# '''
# splitting the datasets using the scikit test-train split method 
# '''

# # x_bc_test = np.vstack([df2[col].values for col in columns])
# # x_bc = x_bc_test.T #final input features
# # y_bc = df["category"]


# Get 80% of the dataset as the training set. Put the remaining 20% in val/testing variables.
X_train, X_val, y_train, y_val = train_test_split(features, category, stratify=category, test_size=0.20, random_state=42)

# print("Shape of df2:", df2.shape)
# print("Shape of X_val:", X_val.shape)


#-- calculating the feature importance -----------------

random_forest_model = RandomForestClassifier(n_estimators = 50,
                                              max_depth = 16, 
                                              min_samples_split = 2).fit(X_train,y_train)
print(f"Metrics train:\n\t Accuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\n Metrics test:\n\t Accuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")


# Use permutation importance based on ROC AUC for the "bad" class
def auc_score(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, 1-prob)  # Class 0 = bad

#auc = auc_score(random_forest_model, X_val, y_val)
#print(f"Model ROC AUC (for predicting 'bad' die, label=0): {auc:.4f}")

perm_importance = permutation_importance(
    random_forest_model, X_val, y_val,
    scoring=auc_score,
    n_repeats=10,
    random_state=42
)



# # Get feature importances
# #importances = random_forest_model.feature_importances_

# # Sort feature importances in descending order
# #indices = np.argsort(importances)[::-1]

# Use SHAP to explain the model's predictions
# explainer = shap.TreeExplainer(random_forest_model)
# shap_values = explainer.shap_values(X_val)[0]

# #print("Shape of shap_values:", shap_values[1].shape)
# #print("Shape of X_val:", X_val.shape)
# # print(shap_values.shape[1])
# # print(X_val.shape[1])

# # Ensure the number of features matches
# #assert shap_values.shape[1] == X_val.shape[1], "Mismatch in number of features between shap_values and X_val"


# # Plot SHAP values for the "burn die" class (assuming class 0 is "burn die")
#shap.summary_plot(shap_values, X_val,plot_type='bar', class_names=["intact die","burnt die"])
                  
# Convert to a DataFrame for plotting
importance_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance': perm_importance.importances_mean
}).sort_values(by='importance', ascending=True)

#importance_df['importance'] = importance_df['importance'].apply(lambda x: x if x > 0 else 0)
importance_df = importance_df[importance_df['importance'] > 0]
# Plot top 20 features
plt.figure(figsize=(10, 6))
#plt.barh(importance_df['feature'][:10][::-1], importance_df['importance'][:10][::-1])
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel("Decrease in AUC when feature is permuted")
plt.title("Permutation Feature Importance (focused on 'bad' die)")
plt.tight_layout()
plt.show()


