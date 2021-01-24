## Import packages
import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
import yaml

from helper_functions import *
from matplotlib.backends.backend_pdf import PdfPages
from pdpbox import pdp
from sklearn.metrics import roc_curve

# # print the JS visualization code to the notebook
# shap.initjs()

# show all numbers in pandas dataframes as 8dp floats
pd.options.display.float_format = '{:,.4f}'.format


class LeafNode:
    def __init__(self, leaf_node_dict, parent_depth=None):
        self.leaf_node_dict = leaf_node_dict
        self.parent_depth = parent_depth
        
        self.leaf_index = self.leaf_node_dict['leaf_index']
        self.leaf_value = self.leaf_node_dict['leaf_value']
        self.leaf_weight = self.leaf_node_dict['leaf_weight']
        self.leaf_count = self.leaf_node_dict['leaf_count']
        self.left_child = None
        self.right_child = None
        self.children = None
        self.node_type = 'leaf'
        
        if not self.parent_depth:
            self.depth = 0
        else:
            self.depth = self.parent_depth + 1
        
        self.max_children = 0

    def __repr__(self):
        return str(
            {
                **self.leaf_node_dict
                , **{
                    'left_child': self.left_child
                    , 'right_child': self.right_child
                    , 'children': self.children
                    , 'node_type': self.node_type
                    , 'depth': self.depth
                    , 'parent_depth': self.parent_depth
                    , 'max_children': self.max_children
                }
            }
        )
    
    def __str__(self):
        return self.__repr__()
    
    def __iter__(self):
        yield self

        
class SplitNode:
    def __init__(self, split_node_dict, features, tree_max_depth, parent_depth=None):
        self.split_node_dict = split_node_dict
        self.features = features
        self.tree_max_depth = tree_max_depth
        self.parent_depth = parent_depth
        
        self.split_index = self.split_node_dict['split_index']
        self.split_feature = self.split_node_dict['split_feature']
        self.split_feature_name = self.features[self.split_feature]
        self.split_gain = self.split_node_dict['split_gain']
        self.threshold = self.split_node_dict['threshold']
        self.decision_type = self.split_node_dict['decision_type']
        self.default_left = self.split_node_dict['default_left']
        self.missing_type = self.split_node_dict['missing_type']
        self.internal_value = self.split_node_dict['internal_value']
        self.internal_weight = self.split_node_dict['internal_weight']
        self.internal_count = self.split_node_dict['internal_count']
        
        self.node_type = 'split'
        
        if self.parent_depth==None:
            self.depth = 0
        else:
            self.depth = self.parent_depth + 1
        
        self.leaf_node_keys = ['leaf_index', 'leaf_value', 'leaf_weight', 'leaf_count']
        
        if any(key in self.leaf_node_keys for key in self.split_node_dict['left_child']):
            self.left_child = LeafNode(self.split_node_dict['left_child'], parent_depth=self.depth)
            self.left_child_type = 'leaf'
        else:
            self.left_child = SplitNode(self.split_node_dict['left_child'], self.features, self.tree_max_depth, parent_depth=self.depth)
            self.left_child_type = 'split'
        
        if any(key in self.leaf_node_keys for key in self.split_node_dict['right_child']):
            self.right_child = LeafNode(self.split_node_dict['right_child'], parent_depth=self.depth)
            self.right_child_type = 'leaf'
        else:
            self.right_child = SplitNode(self.split_node_dict['right_child'], self.features, self.tree_max_depth, parent_depth=self.depth)
            self.right_child_type = 'split'
        

        self.max_children = self.max_children()
    
    def max_children(self):
            left_max_children = self.left_child.max_children if self.left_child else 0
            right_max_children = self.right_child.max_children if self.right_child else 0

            return max(left_max_children, right_max_children) + 1
    
    def __repr__(self):
        return str(
            {
                **self.split_node_dict
                , **{
                    'split_feature_name': self.split_feature_name
                    , 'node_type': self.node_type
                    , 'left_child': self.left_child
                    , 'right_child': self.right_child
                    , 'depth': self.depth
                    , 'parent_depth': self.parent_depth
                    , 'max_children': self.max_children
                    , 'tree_max_depth': self.tree_max_depth
                }
            }
        )
    
    def __str__(self):
        return self.__repr__()
    
    def __iter__(self):
        if self.left_child_type=='split':
            yield self.left_child

        if self.right_child_type=='split':
            yield self.right_child
    
    def update_matrix_for_tree(self, root, matrix, features_ordered_by_gains_filtered, facRow, tree_num):
        
        if root:
            if root.split_feature_name in features_ordered_by_gains_filtered:
                depth = root.depth
                
                # Add the weighted value to the the correct row (facRow[f]), for the current tree (t)
                w = 2**((root.tree_max_depth - root.depth)/2.0)
                matrix[facRow[root.split_feature_name], tree_num, 0] += w
                
                # Add cumulative count
                matrix[facRow[root.split_feature_name], tree_num, 1] += 1
                
                # Add "max depth"
                if matrix[facRow[root.split_feature_name],tree_num,2] == 0 or ((root.tree_max_depth+1) - depth) > matrix[facRow[root.split_feature_name],tree_num,2]: # zero case accounts for first time
                    matrix[facRow[root.split_feature_name],tree_num,2] = (root.tree_max_depth+1) - depth
            
            if root.left_child_type=='split':
                self.update_matrix_for_tree(root.left_child, matrix, features_ordered_by_gains_filtered, facRow, tree_num)
            if root.right_child_type=='split':
                self.update_matrix_for_tree(root.right_child, matrix, features_ordered_by_gains_filtered, facRow, tree_num)
            
            return matrix

        
class Tree:
    def __init__(self, tree_dict, features):
        self.tree_dict = tree_dict
        self.features = features
        
        self.tree_index = self.tree_dict['tree_index']
        self.num_leaves = self.tree_dict['num_leaves']
        self.num_cat = self.tree_dict['num_cat']
        self.shrinkage = self.tree_dict['shrinkage']
        
        self.split_node_keys = ['split_index', 'split_feature', 'split_gain', 'threshold', 'decision_type', 'default_left', 'missing_type', 'internal_value', 'internal_weight', 'internal_count']
        self.leaf_node_keys = ['leaf_index', 'leaf_value', 'leaf_weight', 'leaf_count']
    
        # Initialise tree structure without the correct max depth, so that the tree structure itself can be used to calculate the max_depth
        self.tree_structure = SplitNode(self.tree_dict['tree_structure'], self.features, tree_max_depth=0, parent_depth=None)
        
        # Calculate max_depth and then initialise tree structure with correct max_depth
        self.max_depth = self.max_children()
        self.tree_structure = SplitNode(self.tree_dict['tree_structure'], self.features, tree_max_depth=self.max_depth, parent_depth=None)
        
    
    def max_children(self):
            left_max_children = self.tree_structure.left_child.max_children if self.tree_structure else 0
            right_max_children = self.tree_structure.right_child.max_children if self.tree_structure else 0

            return max(left_max_children, right_max_children) + 1
    
    def __repr__(self):
        return str({**self.tree_dict, **{'max_depth': self.max_depth}})
    
    def __str__(self):
        return self.__repr__()


class Model:
    
    def __init__(self, model, train, validation, test, EVAL_RESULTS, features, categorical_features, primary_constants, timestamp_of_model):
        
        self.model = model
        self.train = train
        self.validation = validation
        self.test = test
        self.EVAL_RESULTS = EVAL_RESULTS
        self.features = features
        self.categorical_features = categorical_features
        self.primary_constants = primary_constants
        self.timestamp_of_model = timestamp_of_model
        
        self.initiate_constants()
    
    
    def initiate_constants(self):
        
        # extract primary constants
        self.user = self.primary_constants['user']
        self.shared_workspace = self.primary_constants['shared_workspace']
        self.user_dir = self.primary_constants['user_dir']
        self.pipeline_dir = self.primary_constants['pipeline_dir']
        self.input_dir = self.primary_constants['input_dir']
        self.output_dir = self.primary_constants['output_dir']
        self.models_store_dir = self.primary_constants['models_store_dir']
        self.seed = self.primary_constants['seed']
        self.response = self.primary_constants['response']
        self.primary_keys = self.primary_constants['primary_keys']
        self.split = self.primary_constants['split']
        self.non_feature_cols = self.primary_constants['non_feature_cols']
        
        # prediction column name
        self.prediction = 'prediction'
        
        # model results directory (model, as well as parameters and feature list are saved here)
        self.model_results_dir = os.path.join(self.models_store_dir, self.timestamp_of_model)
        self.model_parameters_filename = os.path.join(self.model_results_dir, 'model_parameters.json')
        self.model_features_filename = os.path.join(self.model_results_dir, 'model_features.json')
        self.model_categorical_feature_levels_filename = os.path.join(self.model_results_dir, 'categorical_feature_levels.json')
        
        # predictions directory
        self.predictions_dir = os.path.join(self.model_results_dir, 'predictions')
        self.predictions_train_filename = os.path.join(self.predictions_dir, 'predictions_train.csv')
        self.predictions_validation_filename = os.path.join(self.predictions_dir, 'predictions_validation.csv')
        self.predictions_test_filename = os.path.join(self.predictions_dir, 'predictions_test.csv')
        
        # diagnostics directories
        self.model_diagnostics_dir = os.path.join(self.model_results_dir, 'diagnostics')

        ## error curves
        self.error_curves_dir = os.path.join(self.model_diagnostics_dir, 'error_curves')
        self.error_curves_csv_filename = os.path.join(self.error_curves_dir, 'error_curves.csv')
        
        ### LightGBM renames some metrics to aliases. This mapping converts aliases to more human-understandable labels, or leaves them as is if they are already understandable
        self.metric_map = {
            'auc': 'auc'
            , 'binary_logloss': 'binary_logloss'
            , 'l1': 'mean_absolute_error'
            , 'mean_absolute_error': 'mean_absolute_error'
            , 'l2': 'mean_squared_error'
            , 'mean_squared_error': 'mean_squared_error'
            , 'rmse': 'root_mean_squared_error'
            , 'root_mean_squared_error': 'root_mean_squared_error'
        }

        ## pvo
        self.pvo_dir = os.path.join(self.model_diagnostics_dir, 'pvo')

        ## variable importance
        self.variable_importance_dir = os.path.join(self.model_diagnostics_dir, 'variable_importance')
        self.variable_importance_gain_csv_filename = os.path.join(self.variable_importance_dir, 'variable_importance_gain.csv')
        self.variable_importance_split_csv_filename = os.path.join(self.variable_importance_dir, 'variable_importance_split.csv')
        
        ## oneways
        self.oneways_dir = os.path.join(self.model_diagnostics_dir, 'oneways')
        self.oneways_stats_yml_filename = os.path.join(self.oneways_dir, 'oneways_stats.yml')
        self.oneways_response_yml_filename = os.path.join(self.oneways_dir, 'oneways_response.yml')
        self.oneways_prediction_yml_filename = os.path.join(self.oneways_dir,'oneways_prediction.yml')

        ## pdp
        self.pdp_dir = os.path.join(self.model_diagnostics_dir, 'pdp')
        self.numeric_pdps_csv_filename = os.path.join(self.pdp_dir, 'numeric_pdps.csv')

        ## shap
        self.shap_dir = os.path.join(self.model_diagnostics_dir, 'shap')
        self.shap_feature_importance_filename = os.path.join(self.shap_dir, 'shap_feature_importance.csv')

        ## treeviz
        self.treeviz_dir = os.path.join(self.model_diagnostics_dir, 'treeviz')

        # ## other diagnostics
        self.other_diagnostics_dir = os.path.join(self.model_diagnostics_dir, 'other_diagnostics')

        # ROC Curve
        self.roc_curves_dir = os.path.join(self.model_diagnostics_dir, 'roc_curve')

        # Gains curve
        self.gains_curve_dir = os.path.join(self.model_diagnostics_dir, 'gains_curve')
        self.train_validation_gains_curve_csv_filename = os.path.join(self.gains_curve_dir, 'train_validation_gains_curves.csv')

        # Gini
        self.gini_dir = os.path.join(self.model_diagnostics_dir, 'gini')
        self.gini_json_filename = os.path.join(self.gini_dir, 'gini_scores.json')

        # Lift
        self.lift_dir = os.path.join(self.model_diagnostics_dir, 'lift')
        self.train_validation_lift_csv_filename = os.path.join(self.lift_dir, 'train_validation_lift.csv')

        # MAPE
        self.mape_dir = os.path.join(self.model_diagnostics_dir, 'MAPE')

        # Split value histograms
        self.split_value_histograms_dir = os.path.join(self.model_diagnostics_dir, 'split_value_histograms')
        
        # Tree digraphs
        self.tree_digraphs_dir = os.path.join(self.model_diagnostics_dir, 'tree_digraphs')
    
    
    def save_model(self):
        
        print('Saving model...')
    
        self.model_name_gain_txt = 'model_' + self.timestamp_of_model + '_gain.txt'
        self.model_name_split_txt = 'model_' + self.timestamp_of_model + '_split.txt'
        self.model_name_gain_json = 'model_' + self.timestamp_of_model + '_gain.json'
        self.model_name_split_json = 'model_' + self.timestamp_of_model + '_split.json'

        create_dir_if_not_available(self.model_results_dir)

        # Save model as .txt files
        self.model.save_model(filename=os.path.join(self.model_results_dir, self.model_name_gain_txt), importance_type='gain')
        self.model.save_model(filename=os.path.join(self.model_results_dir, self.model_name_split_txt), importance_type='split')

        # Save model as .json files
        self.model_gain_json = self.model.dump_model(importance_type='gain')
        self.model_split_json = self.model.dump_model(importance_type='split')
        save_json(self.model_gain_json, os.path.join(self.model_results_dir, self.model_name_gain_json))
        save_json(self.model_split_json, os.path.join(self.model_results_dir, self.model_name_split_json))
        
        print(f'Saved model at: {self.model_results_dir}' + os.linesep)

    
    def save_model_parameters(self):
        
        print('Saving model parameters...')
    
        # best iteration
        model_best_iteration = self.model.best_iteration

        # best score
        model_best_score = {}
        for key, value in self.model.best_score.items():
            dataset = key
            metrics_and_best_scores = value
            model_best_score[dataset] = list(metrics_and_best_scores.items())

        # params
        model_params = self.model.params

        # num_trees
        model_num_trees = self.model.num_trees()

        # num_features
        model_num_features = self.model.num_feature()


        self.model_parameters_dict = {
            'best_iteration': model_best_iteration
            , 'best_score': model_best_score
            , 'model_params': model_params
            , 'num_trees': model_num_trees
            , 'num_features': model_num_features
        }
        
        # feature list
        model_features = self.model.feature_name()

        self.model_features_dict = {
            'features': {k: v for k,v in enumerate(model_features)}
        }
        
        # list of levels in each categorical feature
        categorical_feature_indices = self.model.params['categorical_column']
        categorical_levels = self.model.pandas_categorical

        self.model_categorical_feature_levels = {
            'categorical_feature_levels': {
                model_features[categorical_feature_index]: {numeric_encoding: level for numeric_encoding, level in enumerate(categorical_levels[categorical_feature_number])} 
                for categorical_feature_number, categorical_feature_index in enumerate(categorical_feature_indices)
            }
        }
        
        # save model parameters
        save_json(self.model_parameters_dict, self.model_parameters_filename)
        save_json(self.model_features_dict, self.model_features_filename)
        save_json(self.model_categorical_feature_levels, self.model_categorical_feature_levels_filename)
        
        print(f'Saved model parameters at: {self.model_parameters_filename}' + os.linesep)

        return self.model_parameters_dict, self.model_features_dict, self.model_categorical_feature_levels
    
    
    def save_predictions(self):
        
        print('Saving predictions...')
        
        self.train[self.prediction] = self.model.predict(self.train[self.features])
        self.validation[self.prediction] = self.model.predict(self.validation[self.features])
        self.test[self.prediction] = self.model.predict(self.test[self.features])

        create_dir_if_not_available(self.predictions_dir)

        self.train[self.primary_keys + [self.prediction]].to_csv(path_or_buf=self.predictions_train_filename, header=True, index=False)
        self.validation[self.primary_keys + [self.prediction]].to_csv(path_or_buf=self.predictions_validation_filename, header=True, index=False)
        self.test[self.primary_keys + [self.prediction]].to_csv(path_or_buf=self.predictions_test_filename, header=True, index=False)
        
        print(f'Saved predictions at: {self.predictions_test_filename}' + os.linesep)
        
        return self.train, self.validation, self.test
    
    
    def create_diagnostics_directories(self):
        
        print('Creating diagnostics directories...')
        
        diagnostics_dir_list = [
            self.model_diagnostics_dir
            , self.error_curves_dir
            , self.pvo_dir
            , self.variable_importance_dir
            , self.oneways_dir
            , self.pdp_dir
            , self.shap_dir
            , self.treeviz_dir
            , self.other_diagnostics_dir
            , self.roc_curves_dir
            , self.gains_curve_dir
            , self.gini_dir
            , self.lift_dir
            , self.mape_dir
            , self.split_value_histograms_dir 
        ]

        for diagnostic_dir in diagnostics_dir_list:
            create_dir_if_not_available(diagnostic_dir)
    
        print(f'Created all diagnostics directories' + os.linesep)
    
    
    def save_error_curves(self):
        
        print('Saving error curves...')
        
        model_metrics = []
        model_metrics_scores = []

        for dataset, metrics_and_scores in self.EVAL_RESULTS.items():
            for metric, scores in metrics_and_scores.items():
                metric_readable = self.metric_map[metric]
                model_metrics.append(dataset + '_' + metric_readable)
                model_metrics_scores.append(scores)
                
                # Save figures
                error_curve_plot = lgb.plot_metric(self.EVAL_RESULTS, metric=metric).get_figure()
                error_curve_plot.savefig(
                    os.path.join(self.error_curves_dir, 'error_curve_' + metric_readable + '.png')
                    , facecolor='white'
                    , edgecolor='white'
                    , transparent=False
                    , bbox_inches = "tight"
                )
        
        # save CSVs
        self.error_curves_data = pd.DataFrame(np.array(model_metrics_scores).T, columns=model_metrics)
        self.error_curves_data.to_csv(path_or_buf=self.error_curves_csv_filename, header=True, index=True, index_label='iteration')
        
        print(f'Saved error curves at: {self.error_curves_dir}' + os.linesep)
        
        return self.error_curves_data
    
    
    def save_variable_importances(self):
        
        print('Saving variable importances...')
        
        def create_variable_importance_df(self, importance_type, importance_col):
            
            variable_importance_data = pd.DataFrame(
                np.array(
                    [
                        self.features
                        , self.model.feature_importance(importance_type=importance_type).tolist()
                    ]
                ).T
                , columns=['feature', importance_col]
            )
            
            variable_importance_data[importance_col] = variable_importance_data[importance_col].astype(float)
            variable_importance_data.sort_values(by=importance_col, axis=0, ascending=False, inplace=True)
            variable_importance_data[importance_col+'_normalised'] = variable_importance_data[importance_col] / variable_importance_data.iloc[0][importance_col]
            
            return variable_importance_data
    
        self.variable_importance_data_gain = create_variable_importance_df(self, 'gain', 'importance_gain')
        self.variable_importance_data_split = create_variable_importance_df(self, 'split', 'importance_split')
        
        # save feature importance order
        self.features_ordered_by_gains = self.variable_importance_data_gain['feature'].tolist()
        self.features_ordered_by_splits = self.variable_importance_data_split['feature'].tolist()
        
        self.variable_importance_data_gain.to_csv(path_or_buf=self.variable_importance_gain_csv_filename, header=True, index=False)
        self.variable_importance_data_split.to_csv(path_or_buf=self.variable_importance_gain_csv_filename, header=True, index=False)
        
        # Plot variable importances
        for importance_type in ['gain', 'split']:
            variable_importance_plot = lgb.plot_importance(
                self.model
                , max_num_features=50
                , importance_type=importance_type
                , precision=2
                , grid=False
                , figsize=(20, 12)
                , height=0.5
                , title='Feature importance (by ' + importance_type + ')'
                , xlabel='Feature importance (' + importance_type + ')'
            )
            variable_importance_plot.get_figure().savefig(
                os.path.join(self.variable_importance_dir, 'variable_importance_' + importance_type + '.png')
                , facecolor='white'
                , edgecolor='white'
                , transparent=False
                , bbox_inches = "tight"
            )
        
        print(f'Saved variable importances at: {self.variable_importance_dir}' + os.linesep)
        
        return self.variable_importance_data_gain, self.variable_importance_data_split

    
    def save_PvOs(self):
        """
        PvO charts are created by plotting the average predicted and observed value for each predicted percentile (i.e. percentiles of the predicted values). 
        They help in understanding:
        - Whether there is any bias at different prediction levels (which can also help to confirm whether your link function is correct in a GLM)
        - The range and distribution of predictions
        - Whether the model has been overfit (but comparing PvO charts created using the train and test / holdout datasets)
        """
        
        print('Saving PvOs...')
        
        # We will use 100 quantiles, i.e. percentiles, for our PvOs
        train = self.train.copy()
        validation = self.validation.copy()
        
        self.num_train_obs = len(np.array(train[self.response]))
        self.num_validation_obs = len(np.array(validation[self.response]))
                
        train['quantile'] = round(self.train[self.prediction].rank(pct=True), 2)
        validation['quantile'] = round(self.validation[self.prediction].rank(pct=True), 2)
        
        def create_pvo_data(self, dataset):
            
            pvo_data = (
                dataset
                .loc[:, ['quantile', self.prediction, self.response]]
                .groupby(
                    'quantile',
                    as_index=False
                )
                .mean()
            )
            
            return pvo_data
        
        # Create PvO data
        self.train_pvo_data = create_pvo_data(self, train)
        self.validation_pvo_data = create_pvo_data(self, validation)
        
        def save_pvo_plot(self, dataset_name, pvo_data, num_obs):
            
            pvo_fig = plot_pvo(
                data=pvo_data,
                title=f'PvO {dataset_name}',
                actual_col=self.response,
                pred_col=self.prediction,
                percentile_col='quantile',
                num_obs=num_obs
            )
            
            pvo_fig.savefig(os.path.join(self.pvo_dir, f'{dataset_name}_PvO.png'), facecolor='white', edgecolor='white', transparent=False, bbox_inches = "tight")
        
        # Save PvO plots
        for dataset_name, pvo_data, num_obs in zip(['train', 'validation'], [self.train_pvo_data, self.validation_pvo_data], [self.num_train_obs, self.num_validation_obs]):
            save_pvo_plot(self, dataset_name, pvo_data, num_obs)
        
        print(f'Saved PvOs at: {self.pvo_dir}' + os.linesep)
        
        return self.train_pvo_data, self.validation_pvo_data
    
    
    def save_oneways(self, num_bins=20, num_cats=20, clip=None, one_ways=True, dashboard=True, normed=False, verbose=True):
        
        print('Building histograms...')
        
        self.d, self.r, self.p = build_hists(
            self.train
            , self.features
            , self.categorical_features
            , response=self.response
            , prediction=self.prediction
            , verbose=verbose
            , num_bins=num_bins
            , num_cats=num_cats
            , clip=clip
        )
    
        y1 = open(self.oneways_stats_yml_filename, "w")
        y1.write(yaml.dump(self.d))
        y1.close()

        y2 = open(self.oneways_response_yml_filename, "w")
        y2.write(yaml.dump(self.r))
        y2.close()

        y3 = open(self.oneways_prediction_yml_filename, "w")
        y3.write(yaml.dump(self.p))
        y3.close()

        print(f'Saved histogram stats at: {self.oneways_stats_yml_filename}')
        print(f'Saved 1-way response at: {self.oneways_response_yml_filename}')
        print(f'Saved 1-way prediction at: {self.oneways_prediction_yml_filename}')
        
        print('Plotting oneways...')
        
        oneway_figs = plot_hists(
            self.oneways_stats_yml_filename
            , self.oneways_response_yml_filename
            , self.oneways_prediction_yml_filename
            , self.features
            , one_ways=one_ways
            , dashboard=dashboard
            , normed=normed
            , verbose=verbose
        )

        # Save all oneways to PDF
        print('Saving oneways...')
        savePDF = PdfPages(os.path.join(self.oneways_dir, 'oneways.pdf'))
        for savefig in oneway_figs:
            savePDF.savefig(savefig)
        savePDF.close()
        
        print(f'Saved oneways at: {self.oneways_dir}' + os.linesep)
        
    
    def save_pdps(self, plot_top_n_features=20, percentile_range=(0,95), num_grid_points=10):
        
        print('Calculating PDPs')
        
        pdps_numeric = []
        for feature in self.features[:plot_top_n_features]:
            if feature not in self.categorical_features:
                print(f'    For {feature}...')
                pdps_numeric.append(
                    (
                        feature
                        , pdp.pdp_isolate(
                            model=self.model,
                            grid_type='percentile',
                            dataset=self.train[~(self.train[feature].isnull())],
                            model_features=self.features,
                            feature=feature,
                            num_grid_points=num_grid_points,
                            percentile_range=percentile_range # we might need to manually craft the percentile ranges for each individual feature to produce ideal pdps
                        )
                    )
                )

        # plot numeric pdps
        print('Saving PDPs...')
        for numeric_pdp_set in pdps_numeric:
            print(f'    For {str(numeric_pdp_set[0])}')
            fig, axes = pdp.pdp_plot(numeric_pdp_set[1], numeric_pdp_set[0], center=False, plot_pts_dist=True)
            fig.savefig(os.path.join(self.pdp_dir, f'{numeric_pdp_set[0]}_pdp.png'))

        numeric_pdp_dataframes = [
            pd.DataFrame(
                {
                    'feature_name': pdp[0],
                    'feature_values': pdp[1].feature_grids,
                    'feature_pdp': pdp[1].pdp,
                }
            )
            for pdp
            in pdps_numeric
        ]
        
        print('Saving PDP csv')
        self.numeric_pdp_dataframe_concatenated = pd.concat(numeric_pdp_dataframes, axis=0)
        self.numeric_pdp_dataframe_concatenated.to_csv(path_or_buf=self.numeric_pdps_csv_filename, header=True, index=False)
        
        return self.numeric_pdp_dataframe_concatenated
    
    
    def save_shap_plots(self, top_n_features_to_plot=20):
        
        print('Saving SHAP plots...')
        
        self.shap_explainer = shap.TreeExplainer(self.model)

        # below shap values are a list of ndarray of length 2. 
        # index 0 = shap values for TARGET=0
        # index 1 = shap values for TARGET=1
        # only difference between values in index 0 and index 1 is the sign of the shap value. I.e. (impact on TARGET=0) = -1 * (impact on TARGET=1)
        print('    Calculating SHAP values...')
        self.shap_values = self.shap_explainer.shap_values(self.train[self.features])
        
        # SHAP summary plots
        print('    Saving SHAP summary plot...')
        shap.summary_plot(self.shap_values, self.train[self.features], class_inds=[1])
        self.shap_summary_plot = plt.gcf()
        self.shap_summary_plot.savefig(
            os.path.join(self.shap_dir, f'shap_summary_plot.png')
            , facecolor='white'
            , edgecolor='white'
            , transparent=False
            , bbox_inches = "tight"
        )
        
        # SHAP variable importances
        print('    Saving SHAP variable importances...')
        self.mean_absolute_shap_vals = np.abs(self.shap_values[1]).mean(axis=0)
        self.shap_feature_importance_pd = pd.DataFrame(list(zip(self.features, self.mean_absolute_shap_vals)), columns=['Feature','Mean absolute shap value'])
        self.shap_feature_importance_pd.sort_values(by=['Mean absolute shap value'], ascending=False,inplace=True)
        self.shap_feature_importance_pd.reset_index(drop=True)
        
        # save shap_feature_importance_pd as csv
        self.shap_feature_importance_pd.to_csv(path_or_buf=self.shap_feature_importance_filename, header=True, index=True, index_label='Rank')
        
        # extract list of top N features by shap impact
        self.features_ordered_by_shap_impact = list(self.shap_feature_importance_pd['Feature'])
        self.top_features_ordered_by_shap_impact = self.features_ordered_by_shap_impact[:top_n_features_to_plot]
        
        # SHAP dependence plots
        ## SHAP dependence plots show the effect of a single feature across the whole dataset.
        ## They plot a feature's value vs. the SHAP value of that feature across many samples.
        ## SHAP dependence plots are scatter plots that show the effect a single feature has on the predictions made by the model.
        ## They are similar to partial dependence plots, but account for the interaction effects present in the features, 
            ## and are only defined in regions of the input space supported by data.
        ## The vertical dispersion of SHAP values at a single feature value is driven by interaction effects, 
            # and another feature is chosen for coloring to highlight possible interactions.
        ## Each dot is a single prediction (row) from the dataset.
        ## The x-axis is the value of the feature.
        ## The y-axis is the SHAP value for that feature, which represents how much knowing that feature's value changes the output of the model for that sample's prediction.
        ## The color corresponds to a second feature that may have an interaction effect with the feature we are plotting (by default this second feature is chosen automatically).
        ## If an interaction effect is present between this other feature and the feature we are plotting it will show up as a distinct vertical pattern of coloring.
        print('    Saving SHAP dependence plots...')
        
        ## first, have to convert levels of categorical variables to int, as dependence plot doesn't work if there are strings in the feature
        self.train_for_shap_dependence = self.train.copy()
        for cat_feature, numeric_encodings_and_level_names in self.model_categorical_feature_levels['categorical_feature_levels'].items():
            mapping_reversed = {v:k for k, v in numeric_encodings_and_level_names.items()}
            self.train_for_shap_dependence[cat_feature] = self.train_for_shap_dependence[cat_feature].replace(mapping_reversed)
            self.train_for_shap_dependence[cat_feature] = self.train_for_shap_dependence[cat_feature].astype('category')
        
        pp = PdfPages(os.path.join(self.shap_dir, 'shap_dependence_plots.pdf'))
        for feature_num, feature in enumerate(list(self.shap_feature_importance_pd['Feature'])[:top_n_features_to_plot]):
            print(f'        Creating SHAP dependence plot number {str(feature_num)}/{str(top_n_features_to_plot)} ({feature})')
            shap.dependence_plot(
                feature
                , self.shap_values[1]
                , self.train_for_shap_dependence[self.features]
                , display_features=self.train[self.features] # display the level names rather than the numerical encodings
                , show=False
            )
            dependence_plot = plt.gcf()
            pp.savefig(dependence_plot)
        pp.close()
        
        print(f'Saved SHAP plots at: {self.shap_dir}' + os.linesep)
        
    
    def save_roc_curves(self):
        """
        ROC charts are typically used in classification type problems, where you are trying to assess a model's ability to class the data into different groups.
        The chart plots the Sensitivity of the model against the False Positive Rate (or 1 - Specificity).
        These values are built based off of a confusion matrix, and are defined as being:
            1 - Specificity = False Positive Rate = FP / (FP + TN)
            Sensitivity = True Positive Rate = TP / (TP + FN)
        In order to build a ROC chart, the points are constructed using many different cut-off thresholds to split the predicted values into Yes / No.
        When the ROC moves towards the upper left, this indicates a better fitting model, with anything sitting below the diagonal line indicating the model is worse than random.
        """
        
        fpr_train, tpr_train, threshold_train = roc_curve(self.train[self.response], self.train[self.prediction])
        fpr_validation, tpr_validation, thresholds_validation = roc_curve(self.validation[self.response], self.validation[self.prediction])
        
        for fpr, tpr, dataset_name in zip([fpr_train, fpr_validation], [tpr_train, tpr_validation], ['Training', 'Validation']):
            fig = plt.figure(figsize=(12,8))
            plt.clf()
            plt.plot(fpr, tpr, label=f'[{dataset_name}] ROC curve', c=qc2, lw=3)
            plt.plot([0,1], [0,1], label='Random model', c='k', lw=3, ls='dashed')
            plt.title(f'[{dataset_name}] ROC curve')
            plt.ylabel('True positive rate (Sensitivity): TP/(TP + FN)')
            plt.xlabel('False positive rate (1 - Specificity): FP/(FP + TN)')
            plt.legend(loc='lower right')
            plt.tight_layout()
            
            fig.savefig(
                os.path.join(self.roc_curves_dir, f'roc_{dataset_name}.png')
                , facecolor='white'
                , edgecolor='white'
                , transparent=False
                , bbox_inches = "tight"
            )
        
        fig = plt.figure(figsize=(12,8))
        plt.clf()
        plt.plot(fpr_train, tpr_train, label=f'[Training] ROC curve', c=qc2, lw=3)
        plt.plot(fpr_validation, tpr_validation, label=f'[Validation] ROC curve', c=qc3, lw=3)
        plt.plot([0,1], [0,1], label='Random model', c='k', lw=3, ls='dashed')
        plt.title(f'ROC curves')
        plt.ylabel('True positive rate (Sensitivity): TP/(TP + FN)')
        plt.xlabel('False positive rate (1 - Specificity): FP/(FP + TN)')
        plt.legend(loc='lower right')
        plt.tight_layout()

        fig.savefig(
            os.path.join(self.roc_curves_dir, f'roc_combined.png')
            , facecolor='white'
            , edgecolor='white'
            , transparent=False
            , bbox_inches = "tight"
        )
        
    
    def save_gains_curves(self):
        """
        Gains curves are used to determine how good a model is at ranking data.
        They represent the percentage of the response that is captured in the highest ranked portion of the data (according to the model).
        A useless model will provide a random prediction which means that in each 10% of the data ranked by the model we will get 10% of the responses.
        The perfect model is determined based on ranking the data by the actual response levels (a perfect ranking).
        The main use of a gains curve is to compare competing models. 
        When comparing multiple models, the model with a gains curve closest to the perfect curve indicates the best model performer.
        """
        
        print('Saving gains curves...')
        
        train = self.train.copy()
        validation = self.validation.copy()
        
        train_gain_curve = plot_cumulative_gain_single(train[self.response], train[self.prediction], title='[Training] Cumulative Gains Curve', figsize=(20, 10))
        validation_gain_curve = plot_cumulative_gain_single(validation[self.response], validation[self.prediction], title='[Validation] Cumulative Gains Curve', figsize=(20, 10))
        
        # Save individual gains curves
        train_gain_curve.savefig(os.path.join(self.gains_curve_dir, 'train_gains_curve.png'), facecolor='white', edgecolor='white', transparent=False, bbox_inches = "tight")
        validation_gain_curve.savefig(os.path.join(self.gains_curve_dir, 'validation_gains_curve.png'), facecolor='white', edgecolor='white', transparent=False, bbox_inches = "tight")
        
        self.gains_data, combined_gains_plot = plot_cumulative_gain_multiple(
            {
                'Training': {
                    self.response: train[self.response]
                    , self.prediction: train[self.prediction]
                }
                , 'Validation': {
                    self.response: validation[self.response]
                    , self.prediction: validation[self.prediction]
                }
            }
            , self.response
            , self.prediction
            , figsize=(20, 10)
        )
        
        # Save combined gains curve data and plot
        self.gains_data.to_csv(path_or_buf=self.train_validation_gains_curve_csv_filename, header=True, index=False)
        combined_gains_plot.savefig(os.path.join(self.gains_curve_dir, 'combined_gains_curve.png'), facecolor='white', edgecolor='white', transparent=False, bbox_inches = "tight")
        
        print(f'Saved gains curves at: {self.gains_curve_dir}' + os.linesep)
        
        return self.gains_data
    
    
    def save_gini_scores(self):
        """
        The gini metric is a measure of how accurately a model sorts the observed values.
        It is dependent only on the order of the predictions, rather than their magnitude.
        The value is between -1 and 1, with a random model giving a value of 0.
        Negative values are not common, and indicate that the model is sorting the data backwards.
        We usually work with the normalised gini, since even a perfect model (where predicted = observed) does not have a gini value of 1
            (the perfect model's gini value depends on the distribution of observed values).
        The normalised gini is calculated by dividing the evaluated model's gini by the perfect model's gini.
        """
        
        print('Saving Gini scores...')
        
        train = self.train.copy()
        validation = self.validation.copy()
        
        self.train_gini = gini_normalised(train[[self.response, self.prediction]], col_obs=self.response, col_pred=self.prediction)
        self.validation_gini = gini_normalised(validation[[self.response, self.prediction]], col_obs=self.response, col_pred=self.prediction)

        gini_scores = {
            'train': self.train_gini
            , 'validation': self.validation_gini
        }
        
        save_json(gini_scores, self.gini_json_filename)
        
        print(f'Saved Gini scores at: {self.gini_json_filename}' + os.linesep)
        
        return self.train_gini, self.validation_gini
    
    
    def save_lift_plots(self, train_gini, validation_gini):
        """
        Lift charts show how well a model is at splitting/separating the high and low observed responses.
        They are particularly useful when you are trying to define a segment using the model, and need to consider the trade-off between segment size and model uplift.
        
        The chart shows cumulative lift.
        Cumulative lift is created by taking the observed average for the top X percentiles and dividing by the overall observed average.
        For this reason it will always converge to 1 at the tail.
        They can be interpreted as showing the increase in response rate from the model compared to taking a simple random sample.
        """
        
        print('Saving lift plots...')
        
        train = self.train.copy()
        validation = self.validation.copy()
        
        train_lift_plot, train_lift_data, train_cum_lift_data = plot_lift(train[[self.response, self.prediction]], col_obs=self.response, col_pred=self.prediction, gini=True, return_data=True)
        validation_lift_plot, validation_lift_data, validation_cum_lift_data = plot_lift(validation[[self.response, self.prediction]], col_obs=self.response, col_pred=self.prediction, gini=True, return_data=True)

        # create lift dataframe
        self.train_validation_lift_data = pd.DataFrame(
            np.array(
                [
                    np.arange(100, 0, -1)
                    , train_lift_data
                    , validation_lift_data
                    , train_cum_lift_data
                    , validation_cum_lift_data
                ]
            ).T
            , columns=['Percentile', 'Train Lift', 'Validation Lift', 'Train Cumulative Lift', 'Validation Cumulative Lift']
        )
        
        # save csv
        self.train_validation_lift_data.to_csv(path_or_buf=self.train_validation_lift_csv_filename, header=True, index=False)

        # save individual lift curves
        train_lift_plot.savefig(os.path.join(self.lift_dir, 'train_lift.png'), facecolor='white', edgecolor='white', transparent=False, bbox_inches = "tight")
        validation_lift_plot.savefig(os.path.join(self.lift_dir, 'validation_lift.png'), facecolor='white', edgecolor='white', transparent=False, bbox_inches = "tight")

        # save combined lift curve
        (
            plot_combined_lift(self.train_validation_lift_data, train_gini=train_gini, validation_gini=validation_gini)
            .savefig(os.path.join(self.lift_dir, 'combined_lift.png'), facecolor='white', edgecolor='white', transparent=False, bbox_inches = "tight")
        )
        
        print(f'Saved lift plots at: {self.lift_dir}' + os.linesep)
        
        return self.train_validation_lift_data
    
    
    def save_split_value_histograms(self, top_n_features_to_plot=50):
        
        print('Saving split value histograms...')
        
        # only plot split value histogram of numeric features in the top 50 features (by gains)
        for feature in self.features_ordered_by_gains[:top_n_features_to_plot]:
            
            if feature not in self.categorical_features: # only available for numeric features
                
                print(f'    For: {feature}...')
                
                split_value_histogram = lgb.plot_split_value_histogram(self.model, feature=feature, figsize=(15,10))
                
                # save fig
                split_value_histogram.get_figure().savefig(
                    os.path.join(self.split_value_histograms_dir, 'split_value_histogram_' + feature + '.png')
                    , facecolor='white'
                    , edgecolor='white'
                    , transparent=False
                    , bbox_inches = "tight"
                )
        
        num_numeric_features_in_top50_features_by_gains = len([feature for feature in self.features_ordered_by_gains[:top_n_features_to_plot] if feature not in self.categorical_features])
        
        print(f'Saved {num_numeric_features_in_top50_features_by_gains} split value histograms at: {self.split_value_histograms_dir}' + os.linesep)

        
    def save_treeviz(self, max_features=50, cmap='inferno'):
        
        print('Saving treeviz plots...')
        
        list_of_tree_infos = self.model_gain_json['tree_info']
        n_trees = len(list_of_tree_infos)

        # Cull to top n features only
        if len(self.features_ordered_by_gains) < max_features:
            max_features = len(self.features_ordered_by_gains)
        else:
            max_features = max_features
        features_ordered_by_gains_filtered = self.features_ordered_by_gains[0:max_features]

        # Assign some row IDs (into a dictionary) to each factor, based on importance rank
        facRow = {}
        for i in range(max_features):
            facRow[features_ordered_by_gains_filtered[i]] = i
        
        # Create a matrix to store the data in
        matrix = np.zeros((max_features, n_trees, 3))
        
        for t in range(n_trees):
            tree = Tree(list_of_tree_infos[t], self.features)
            root = tree.tree_structure

            matrix = root.update_matrix_for_tree(root, matrix, features_ordered_by_gains_filtered, facRow, t)
        
        # Plot
        plt.close("all")

        # Mask the array from NaNs
        mm = matrix.copy()
        mm[mm==0] = np.nan
        mmm = np.ma.array(mm, mask=np.isnan(mm))

        # Set this to something else to show where the factor wasn't actually used
        # This is the form (colour, alpha)
        # So setting it to ('k', 0.5') shows black at 0.5 opacity for unused blocks
        #cmap.set_bad('k', 1) 

        #Get the user default colourmap. If it's the default, then edit it to show black for when
        #the factor isn't used, rather than white
        cmap = 'inferno'
        my_cmap = mpl.cm.get_cmap(cmap)
        if cmap == "inferno":
            my_cmap.set_bad('k',1)

        # Produce the 3 plots
        fig1 = plt.figure(figsize=(20,10))
        plt.clf()
        plt.imshow(mmm[:,:,0], aspect="auto", interpolation="nearest", cmap=cmap)
        plt.xlabel("Tree number")
        plt.yticks(range(max_features), features_ordered_by_gains_filtered)
        plt.colorbar().set_label("weighted value")
        plt.tight_layout()

        fig2 = plt.figure(figsize=(20,10))
        plt.clf()
        plt.imshow(mmm[:,:,1], aspect="auto", interpolation="nearest", cmap=cmap)
        plt.xlabel("Tree number")
        plt.yticks(range(max_features), features_ordered_by_gains_filtered)
        plt.colorbar().set_label("cumulative count")
        plt.tight_layout()

        fig3 = plt.figure(figsize=(20,10))
        plt.clf()
        plt.imshow(mmm[:,:,2], aspect="auto", interpolation="nearest", cmap=cmap)
        plt.xlabel("Tree number")
        plt.yticks(range(max_features), features_ordered_by_gains_filtered)
        plt.colorbar().set_label("max depth")
        plt.tight_layout()
        
        # save fig objects
        output_file = os.path.join(self.treeviz_dir, 'treeviz_plots.pdf')
        figs = [fig1, fig2, fig3]
        pp = PdfPages(output_file)
        for fig in figs:
            pp.savefig(fig)
        pp.close()
        
        print(f'Saved treeviz plots at: ' + os.linesep)
