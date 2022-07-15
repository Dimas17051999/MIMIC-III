#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import string
import pickle
import warnings
import numpy as np
import pandas as pd
from pandas import read_csv
warnings.filterwarnings('ignore')

#Preprocessing
from sklearn import utils
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline

#Models
#!pip install xgboost
import xgboost
from sklearn import svm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, accuracy_score, f1_score

##plotting libraries
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn import tree
import matplotlib.pylab as pl
#import shap
from sklearn.model_selection import train_test_split


# In[ ]:





# In[2]:




#----------- Hyperparameter
cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=422) #for training
#--------------------------

def read_data(raw_clinical_note):
    """ Read clinical data """
    data = pd.read_csv(raw_clinical_note, header=0,na_filter=True)
    col = data.columns
    x = data.drop('insulin_short', axis = 1) #features
    y = data.insulin_short.to_numpy() #label
    feature_list = list(x.columns)
    return x, y, feature_list

def generating_metrics(model, model_ehr, x, y):
    """Function to generate metrics: auc_score, sensitivity, specificity, f1, accuracy"""
    if model == "LR" or model =="RF" or model =="ADA" or model =="GBT" or model =="XGBT" or model =="LightGB":
        y_pred_proba = model_ehr.predict_proba(x)[:, 1]
        y_pred = model_ehr.predict(x)
        y_predicted = np.where(y_pred > 0.5, 1, 0) #Turn probability to 0-1 binary output
        acc = accuracy_score(y,y_predicted)
        tn, fp, fn, tp = confusion_matrix(y,y_predicted).ravel()
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_pred_proba)
    else:
        y_pred = model_ehr.predict(x)
        y_predicted = np.where(y_pred > 0.5, 1, 0) #Turn probability to 0-1 binary output
        acc = accuracy_score(y,y_predicted)
        tn, fp, fn, tp = confusion_matrix(y,y_predicted).ravel()
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_predicted)

    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    auc_score = auc(false_positive_rate, true_positive_rate)
    f1 = f1_score(y, y_predicted)
    return auc_score, sensitivity, specificity, f1, acc, false_positive_rate, true_positive_rate

def saving_metrics(model_name, logs_file, num_features, auc_train
                   ,auc_val, sens_val, spec_val, f1_val, acc_val
                   ,auc_test, sens_test, spec_test, f1_test, acc_test,fpr, tpr):
    """ Saving final metrics in csv file.
    Metrics generated during training, validation, testing steps are saved"""
    name = pd.DataFrame({'model_name':model_name}, index=[0])
    num_features = pd.DataFrame({'num_features':num_features}, index=[0])
    auc_train = pd.DataFrame({'auc_train':auc_train},index = [0])
    auc_val = pd.DataFrame({'auc_val':auc_val},index = [0])
    sens_val = pd.DataFrame({'sens_val':sens_val},index = [0])
    spec_val = pd.DataFrame({'spec_val':spec_val},index = [0])
    f1_val = pd.DataFrame({'f1_val':f1_val},index = [0])
    acc_val = pd.DataFrame({'acc_val':acc_val},index = [0])
    auc_test = pd.DataFrame({'auc_test':auc_test},index = [0])
    sens_test = pd.DataFrame({'sens_test':sens_test},index = [0])
    spec_test = pd.DataFrame({'spec_test':spec_test},index = [0])
    f1_test = pd.DataFrame({'f1_test':f1_test},index = [0])
    acc_test = pd.DataFrame({'acc_test':acc_test},index = [0])

    fpr = str(fpr)
    tpr = str(tpr)
    fpr = pd.DataFrame({'false_positive_rate':fpr},index = [0])
    tpr = pd.DataFrame({'true_positive_rate':tpr},index = [0])

    frames = [name, num_features, auc_train, auc_val,sens_val,spec_val,f1_val,acc_val,
              auc_test,sens_test,spec_test,f1_test,acc_test, fpr, tpr]
    resultado = pd.concat(frames, axis = 1)
    url_log = model_name +'_metrics.csv'
    url_log = os.path.join(logs_file,str(url_log))
    resultado.to_csv(url_log)

def create_folder(logs_file):
    try:
        if not os.path.exists(logs_file):
            os.makedirs(logs_file)
    except Exception as e:
        raise

def saving_parameters(num_features, best_params, auc_training, auc_validation, model_name,logs_file):
    """ Once that fine-tuning was done, the best parameters are saved"""
    name = pd.DataFrame({'model_name':model_name}, index=[0])
    num_features = pd.DataFrame({'num_features':num_features}, index=[0])
    auc_training = pd.DataFrame({'auc_training': auc_training}, index = [0])
    auc_validation = pd.DataFrame({'auc_validation': auc_validation}, index = [0])
    best_params = pd.DataFrame({'best_params': best_params})
    frames = [name, auc_training, auc_validation, best_params]
    resultado = pd.concat(frames, axis = 1)
    output_file = model_name +'_parameters.csv'
    output_file = os.path.join(logs_file,str(output_file))
    resultado.to_csv(output_file)

def imputer(set):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    set = imputer.fit_transform(set)
    return set

def scaler(set):
    scaler = StandardScaler()
    set = scaler.fit_transform(set)
    return set

def features_selection(x_train, y_train,x_val,x_test,model,feature_list):
    """Feature ranking with recursive feature elimination using pipeline"""
    n_features = x_train.shape[1]
    print("n_features original: ",n_features)
    if model == 'LR':
        estimator = LogisticRegression(random_state = 442, penalty = 'elasticnet', solver= 'saga',l1_ratio=0.5)
    if model == 'SVM':
        estimator = svm.LinearSVC(class_weight = 'balanced', random_state = 442)
    if model == 'SGD':
        estimator = SGDClassifier(class_weight = 'balanced', random_state = 442)
    if model == 'ADA':
        estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, class_weight = 'balanced'),random_state = 442)
    if model == 'RF':
        estimator = RandomForestClassifier(random_state=442, class_weight = 'balanced')
    if model == 'GBT':
        estimator = GradientBoostingClassifier(random_state = 442)
    if model == 'XGBT':
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train==1)
        estimator = XGBClassifier(seed = 442,eval_metric = 'auc', scale_pos_weight = ratio)
    if model == 'LightGB':
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train==1)
        estimator = lgb.LGBMClassifier(seed = 442, scale_pos_weight = ratio)

    print("Searching RFE")
    classifier = RFE(estimator=estimator, step=1)
    model = Pipeline([('classifier', classifier)])
    parameters = {'classifier__n_features_to_select': [int(n_features*0.25),int(n_features*0.5),int(n_features*0.75),n_features]}
    grid = GridSearchCV(model, parameters, cv=3)
    grid.fit(x_train, y_train)
    num_features = grid.best_params_
    num_features = re.sub(r'[^\d]','',str(num_features))
    print("Optimal number of features",num_features)

    print("SelectKBest")
    selector = SelectKBest(f_classif, k=int(num_features)) #we pass the "optimal number of features" discovered in the previous pass
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    feature_list = [feature_list[i] for i in selector.get_support(indices=True)]
    return x_train, x_val, x_test,feature_list, num_features

def mortality_model(train):
    """===================== Loading data ================================================================"""
    #create_folder(logs_file)
    x_train, y_train, feature_list = read_data(train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
    #x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.33, random_state=42)
    #create_folder(logs_file)
    x_train = scaler(x_train)
    #x_val = scaler(x_val)
    #x_test = scaler(x_test)
    model = 'XGBT'
    """ Imbalanced classes """
    sample_weights = class_weight.compute_sample_weight('balanced', y_train)
    print("=====================  Fine-tuning  ==============================================================")
    if model == 'LR':
        x_train = (x_train-x_train.mean())/(x_train.max()-x_train.min())
        parameters={"C":np.logspace(-3,3,7), "penalty":["elasticnet"],"solver":['saga'], "l1_ratio":[0.5],
                    "class_weight": ['balanced'],}
        estimator = LogisticRegression()
    if model == 'XGBT':
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train==1)
        parameters={"n_estimators":[100],
                    "objective": ['reg:logistic'], "eval_metric": ['auc']}
        estimator = xgb.XGBClassifier()
    print("-----------GridSearchCV-----------------")
    grid = GridSearchCV(estimator=estimator, param_grid=parameters, cv = cv, scoring='roc_auc', refit = True)
    grid.fit(x_train,y_train,sample_weight = sample_weights)
    auc_train = grid.best_score_
    best_params = grid.best_params_

    print("===================== Training again with best parameters =========================================")
    if model == "LR":
        model_ehr = LogisticRegression(**best_params)
        model_ehr = model_ehr.fit(x_train,y_train)
    if model == "XGBT":
        model_ehr = xgb.XGBClassifier(**best_params)
        model_ehr = model_ehr.fit(x_train,y_train)

    """ Saving metrics"""
    auc_val, sens_val, spec_val, f1_val, acc_val,_,_ = generating_metrics(model, model_ehr, x_val, y_val) #val_set
    #auc_test, sens_test, spec_test, f1_test, acc_test,fpr, tpr = generating_metrics(model, model_ehr, x_test, y_test) #test_set
    print("auc_train:{},  auc_val:{}, sens_val {}, spec_val: {}, f1_val {}, acc_val {}".format(auc_train,
                                                                                                    auc_val, sens_val, spec_val, f1_val, acc_val))
    print(best_params)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

mortality_model("C:/Users/Fernando/Documents/SSA/CSV_Tabular/datos_ready_prueba.csv")


# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[15]:


from pandas import read_csv

dataframe = read_csv("C:/Users/Fernando/Documents/SSA/CSV_Tabular/datos_ready_prueba.csv")
dataframe


# In[18]:


dataframe=dataframe.drop(dataframe.columns[0:2], axis=1)

dataframe


# In[19]:


data = dataframe.values
X, y = data[:, :-1], data[:, -1]


# In[32]:


dataframe.to_csv(path_or_buf="C:/Users/Fernando/Documents/SSA/CSV_Tabular/datos_ready_prueba2.csv", index=False)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13)

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.01,
          'loss': 'ls'}


# In[23]:


reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[ ]:





# In[28]:


dataframe.columns


# In[29]:


feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(dataframe.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(dataframe.columns)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


# In[33]:


np.array(dataframe.columns)[sorted_idx]


# In[34]:


get_ipython().system('pip install shap')


# In[35]:


import shap
shap.initjs()


# In[38]:


explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values,X,
                    feature_names=dataframe.columns,
                    plot_type="dot",
                    max_display=20)


# In[39]:


explainer = shap.TreeExplainer(reg)

shap_values = explainer.shap_values(X)
    
feature_list=dataframe.columns.tolist()

shap.dependence_plot(feature_list.index('age'), 
                         shap_values, X, 
                         interaction_index=feature_list.index('glucose_max'),
                         feature_names=feature_list)


# In[ ]:




