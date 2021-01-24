# Imports
import getpass
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import time

from classes import Model
from datetime import datetime
from helper_functions import *
from q_colours import *

# show all numbers in pandas dataframes as 4dp floats
pd.options.display.float_format = '{:,.4f}'.format


# primary constants
user = getpass.getuser()
shared_workspace = '/home/nikankarla/modelling_club_team9'
user_dir = os.path.join(shared_workspace, user)
pipeline_dir = os.path.join(user_dir, 'pipeline')
input_dir = os.path.join(pipeline_dir, 'input')
output_dir = os.path.join(pipeline_dir, 'output')
models_store_dir = output_dir # "models_store_dir" is used in several locations for legacy reasons, so renaming output_dir as models_store_dir 

seed = 2021
response = 'TARGET'
primary_keys = ['SK_ID_CURR']
split = 'split'
non_feature_cols = primary_keys + [response] + [split]

primary_constants = {
    'user': user
    , 'shared_workspace': shared_workspace
    , 'user_dir': user_dir
    , 'pipeline_dir': pipeline_dir
    , 'input_dir': input_dir
    , 'output_dir': output_dir
    , 'models_store_dir': models_store_dir
    , 'seed': seed
    , 'response': response
    , 'primary_keys': primary_keys
    , 'split': split
    , 'non_feature_cols': non_feature_cols
}



### Define input datasets. Once we start using model files directly from input dir, read in with: model_file = pd.read_csv('...')
home_loan_train = pd.read_csv(os.path.join(shared_workspace, 'git_repos/mc2020_team9/data/raw/application_train.csv'))
home_loan_test = pd.read_csv(os.path.join(shared_workspace, 'git_repos/mc2020_team9/data/raw/application_test_noTarget.csv'))
home_loan_test[response] = None # Placeholder, target of test data should already be set to None in the data preparation step

## Merge train and test into one table. NOTE: not needed once we start using model files directly from input dir
home_loan_train['split'] = np.random.choice(
    a=["train", "validation"],
    p=[0.75, 0.25],
    size=home_loan_train.shape[0]
)
home_loan_test['split'] = "test"
model_file = pd.concat([home_loan_train, home_loan_test], axis=0)

# keep this line
model_file[response] = model_file[response].astype(float)

# feature list
features = [col for col in model_file.columns if col not in non_feature_cols]

# convert categorical features to "category" type
columns_and_types = model_file.dtypes.to_dict()
categorical_features = [col for col, typ in columns_and_types.items() if (typ.name in ['object', 'category']) and col in features]
for col in categorical_features:
    model_file[col] = model_file[col].astype('category')
    

# need to add a step here to inform the model that some numeric columns (e.g. flag columns) should actually by considered as categorical...


# split model file into train/validation/test
train = model_file.loc[model_file['split'] == "train"]
validation = model_file.loc[model_file['split'] == "validation"]
test = model_file.loc[model_file['split'] == "test"]

train.drop(columns=['split'], inplace=True)
validation.drop(columns=['split'], inplace=True)
test.drop(columns=['split'], inplace=True)


# convert splits into lightgbm datasets
train_dataset = lgb.Dataset(
    data=train[features],
    label=train[response],
    feature_name=features, 
    categorical_feature=categorical_features,
    free_raw_data=False
)
validation_dataset = lgb.Dataset(
    data=validation[features],
    label=validation[response],
    feature_name=features, 
    categorical_feature=categorical_features,
    reference=train_dataset,
    free_raw_data=False
)


# train model
EVAL_RESULTS = {}
params = {
    'objective': 'binary',  # binary = log loss objective function
    'metric': ['auc', 'binary_logloss', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error'],
    'first_metric_only': True,
    'n_jobs': -1,
    'learning_rate': 0.1,
    "seed": seed
}

model_temp = lgb.train(
    params=params,
    train_set=train_dataset,
    num_boost_round=2500,
    early_stopping_rounds=25,
    valid_sets=[train_dataset, validation_dataset],
    verbose_eval=1,
    evals_result=EVAL_RESULTS
)


timestamp_of_model = datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
model = Model(
    model=model_temp
    , train=train
    , validation=validation
    , test=test
    , EVAL_RESULTS=EVAL_RESULTS
    , features=features
    , categorical_features=categorical_features
    , primary_constants=primary_constants
    , timestamp_of_model=timestamp_of_model
)


model.save_model()
model_parameters_dict, model_features_dict, model_categorical_feature_levels = model.save_model_parameters()
# add method here to save scores to scoreboard, for tracking
train, validation, test = model.save_predictions()
model.create_diagnostics_directories()
error_curves_data = model.save_error_curves()
variable_importance_data_gain, variable_importance_data_split = model.save_variable_importances()
train_pvo_data, validation_pvo_data = model.save_PvOs()

# NOTE: we are clipping to the 95th percentile to ignore outliers for the purposes of oneways
model.save_oneways(clip=(0, 0.95))

# NOTE: we are only considering up to the 95th percentile for the purposes of pdps
numeric_pdps = model.save_pdps(plot_top_n_features=20, percentile_range=(0,95), num_grid_points=10)

model.save_shap_plots(top_n_features_to_plot=20)
model.save_roc_curves()
gains_data = model.save_gains_curves()
train_gini, validation_gini = model.save_gini_scores()
train_validation_lift_data = model.save_lift_plots(train_gini, validation_gini)
model.save_split_value_histograms(top_n_features_to_plot=50)
model.save_treeviz(max_features=50)

# # Visualise shap impacts for a single prediction (E.g. for 5th observation, change "row" to 5)
# shap.decision_plot(model.explainer.expected_value[1], model.shap_values_train[1][row,:], model.validation[features].iloc[row,:], feature_display_range=slice(None, -31, -1))

# # Visualise a tree. ****Only works in Jupyter****
# lgb.create_tree_digraph(model.model, tree_index=1, show_info='split_gain', orientation='vertical')

