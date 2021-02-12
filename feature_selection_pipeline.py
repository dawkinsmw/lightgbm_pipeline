## Import packages
import gc
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import shap
import sys
import time

from datetime import datetime
from functools import reduce

# print the JS visualization code to the notebook
shap.initjs()

# show all rows and columns of pandas dataframe
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 3600)

# show all numbers in pandas dataframes as 8dp floats
pd.options.display.float_format = '{:,.4f}'.format


user = os.getenv('USERNAME')
shared_workspace = '/home/mdawkins/modelling_club'
user_dir = os.path.join(shared_workspace, user)
input_dir = shared_workspace + "/pipeline/input"
output_dir = shared_workspace + "/pipeline/output"


seed = 2021
response = 'TARGET'
primary_keys = ['SK_ID_CURR']
split = 'SPLIT'
non_feature_cols = primary_keys + [response] + [split]


# Read in train and test data here
model_file = pd.read_pickle(input_dir + "/model_file.pkl")


def model_train(model_file):

    features = [col for col in model_file.columns if col not in non_feature_cols]
    
    columns_and_types = model_file.dtypes.to_dict()
    categorical_features = [col for col, typ in columns_and_types.items() if (typ.name in ['object', 'category']) and col in features]
    for col in categorical_features:
        model_file[col] = model_file[col].astype('category')
    
    train = model_file.loc[model_file[split] == "train"]
    validation = model_file.loc[model_file[split] == "validation"]
    test = model_file.loc[model_file[split].isnull()]

    train.drop(columns=[split], inplace=True)
    validation.drop(columns=[split], inplace=True)
    test.drop(columns=[split], inplace=True)
    
    
    train_dataset = lgb.Dataset(
    data=train[features],
    label=train[response],
    feature_name=features,
    params={'verbose': -1},    
    categorical_feature=categorical_features
    )
    validation_dataset = lgb.Dataset(
        data=validation[features],
        label=validation[response],
        feature_name=features, 
        params={'verbose': -1},    
        categorical_feature=categorical_features,
        reference=train_dataset
    )
    EVAL_RESULTS = {}
    params = {
    'objective': 'binary',  # binary = log loss objective function
    'metric': 'auc',
    'n_jobs': -1,
    'learning_rate': 0.1,
    "seed": seed,
    'verbose': -1
    }

    model = lgb.train(
        params=params,
        train_set=train_dataset,
        num_boost_round=2500,
        early_stopping_rounds=25,
        valid_sets=[train_dataset, validation_dataset],
        verbose_eval=False,
        evals_result=EVAL_RESULTS
    )
    return model.best_score['valid_1']['auc']


# feature list
included_features = ["TARGET","SPLIT","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]
potential_features = [col for col in model_file.columns if (col not in included_features) and (col not in ["SK_ID_CURR","CODE_GENDER",'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'])]

# convert categorical features to "category" type
def forward_step(step_inc_features,step_pot_features):
    # 
    step_scores={}
    
    for step_pot_feat in step_pot_features:
        step_model_features = step_inc_features + [step_pot_feat]
        step_scores[step_pot_feat]= model_train(model_file[step_model_features])
#         print("Completed:" + str(step_pot_feat))
    
    step_best_feature = max(step_scores, key=step_scores.get)
    step_best_auc =  step_scores[step_best_feature]
    print("STEP BEST FEATURE" + str((step_best_feature, step_best_auc)))
    return (step_best_feature, step_best_auc)
    

def forward_feed(model_file,inc_features,pot_features, base_score):  
    best_feature,best_auc = forward_step(inc_features,pot_features)
    with open(output_dir+'/forward_feed_scores.csv','a') as ffs:
        ffs.write("\n" + best_feature +","+str(best_auc))
        
    if ((pot_features == []) or (best_auc < base_score)):
        return inc_features
    
    else:   
        inc_features = forward_feed(model_file, inc_features + [best_feature], [x for x in pot_features if x!=best_feature], best_auc)
        return inc_features