import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml

from q_colours import *


def create_dir_if_not_available(dir_path):
    if not os.path.exists(dir_path):
        print(f'Created directory: {dir_path}')
        os.makedirs(dir_path)
    else:
        print(f'Directory already exists: {dir_path}')


def save_json(dictionary, file_name_including_file_path_and_file_type):
    
    create_dir_if_not_available(os.path.dirname(os.path.realpath(file_name_including_file_path_and_file_type)))
    
    with open(file_name_including_file_path_and_file_type, "w") as save_file:
        json.dump(dictionary, save_file, indent=4)
        
    print(f'Saved: {file_name_including_file_path_and_file_type}')


def plot_pvo(data, title="PvO: overall", percentile_col='ntile', actual_col='response', pred_col='predictions', num_obs=None, log_scale=False):
    
    data = data.sort_values(percentile_col, ascending=True)
    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    percentile = 100.0 * np.array(data[percentile_col])
    avg_predicted = data[pred_col]
    avg_actual = data[actual_col]

    # Main PvO
    plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.plot(percentile, avg_predicted, c=qc3, lw=3, label="Predicted")
    plt.plot(percentile, avg_actual, c=qc2, lw=3, label="Observed")
    plt.legend(loc="upper left")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.title(title + os.linesep + 'Num obs: ' + str(num_obs))
    if log_scale:
        plt.yscale("log")
        deltadiff = np.log(avg_predicted / avg_actual)
        diff_ylabel = "Pred/Obs (log)"
    else:
        deltadiff = avg_predicted - avg_actual
        diff_ylabel = "Pred - Obs"

    # Delta
    plt.subplot2grid((3, 1), (2, 0))
    plt.plot(percentile, deltadiff, c=qc1, lw=3)
    plt.axhline(y=0.0, color='r', linestyle='-')
    plt.xlabel("Percentile (as ranked by the model)")
    plt.ylabel(diff_ylabel)
    plt.ylim(top=0.15, bottom=-0.15)
    plt.grid(True)
    plt.tight_layout()

    return fig


def build_hists(dataset, features, categorical_features, verbose=True, response='TARGET', prediction='prediction', num_bins=20, num_cats=20, clip=None):
    """
    clip = tuple of quantiles to cap histogram ranges at
    num_cats = number of categorical levels to consider in histogram
    """ 
    
    d, r, p = {}, {}, {}
    d['hist_clippings'] = clip
    
    for feature_idx, fname in enumerate(features):
        dd = {}
        if verbose:
            print("Building hist for", fname)
        fcol = dataset[fname]
        total_len = len(fcol) * 1.0
        fcat = fname in categorical_features # flag if feature is categorical
        
        # Numeric
        if not fcat:
            # Stats
            fmin, fmax, fmean, fmed, fmod = fcol.min(), fcol.max(), fcol.mean(), fcol.median(), 0
            fnull, fzero = fcol.isnull().sum()/total_len, len(fcol[fcol==0])/total_len
            fcol_nonnull = fcol[~fcol.isnull()]
            
            # Set range for hists
            if (clip == None): hist_range = [fmin, fmax]
            else: hist_range = [fcol.quantile(clip[0]), fcol.quantile(clip[1])] # Can be less verbose in pandas update
            
            # Get the stats
            fcounts, fbins = np.histogram(fcol_nonnull, bins=num_bins, range=hist_range)
            fcounts, fbins = list(fcounts), list(fbins)
            fbin_centres = list(fbins[:-1] + np.diff(fbins)/2.)
            fbin_centres = [float(x) for x in fbin_centres]
            
            # Response stuff
            pd_data2 = dataset[[fname, response, prediction]]
            
            # Get average response and prediction when feature is NULL
            pd_data2["isnull"] = fcol.isnull()
            try: rnull = float(pd_data2.groupby("isnull").agg("mean")[response].loc[True])
            except: rnull = 0
            try: pnull = float(pd_data2.groupby("isnull").agg("mean")[prediction].loc[True])
            except: pnull = 0
            
            # Get average response and predition when feature is ZERO
            pd_data2["iszero"] = (fcol==0)
            pd_data_groupzero = pd_data2.groupby("iszero").agg("mean")
            try: rzero = float(pd_data_groupzero[response].loc[True])
            except: rzero = 0
            try: pzero = float(pd_data_groupzero[prediction].loc[True])
            except: pzero = 0
            
            # Get average response and prediction when feature is NONZERO
            try: rnonzero = float(pd_data_groupzero[response].loc[False])
            except: rnonzero = 0
            try: pnonzero = float(pd_data_groupzero[prediction].loc[False])
            except: pnonzero = 0
            
            # Get average response and prediction for non-NULL feature values, averaged over each histogram bin
            pd_data_nonnull = pd_data2[~fcol.isnull()]
            pd_data_nonnull.reset_index(inplace=True)
            fbins2 = list(np.asarray(fbins).copy())
            pd_data_nonnull["bin_ids"] = pd.Series(np.digitize(pd_data_nonnull[fname].values, fbins2))
            pd_data_nonnull["bin_ids"][pd_data_nonnull["bin_ids"]==0] = 1
            pd_data_nonnull["bin_ids"][pd_data_nonnull["bin_ids"]==num_bins+1] = num_bins
            pd_data_nonnull_grp = pd_data_nonnull.groupby("bin_ids").agg("mean")
            rbins = list(pd_data_nonnull_grp[response].values)
            rbin_ids = list(pd_data_nonnull_grp.index.values)
            pbins = list(pd_data_nonnull_grp[prediction].values)
            pbin_ids = list(pd_data_nonnull_grp.index.values)
        
        # Categorical
        else: 
            # Stats
            fnull, fzero = fcol.isnull().sum()/total_len, len(fcol[fcol==0])/total_len
            pd_fcounts = fcol.value_counts()
            if len(pd_fcounts) < num_cats: fnumcats = len(pd_fcounts)
            else: fnumcats = num_cats
            num_levels = len(pd_fcounts)
            pd_fcounts = pd_fcounts.iloc[0:fnumcats]
            fbin_centres, fcounts = list(pd_fcounts.index), list(pd_fcounts.values)
            fmin, fmax, fmean, fmed, fmod = 0, num_levels, 0, 0, pd_fcounts.index[0]

            # Response stuff
            pd_data2 = dataset[[fname, response, prediction]]
            pd_data2["isnull"] = fcol.isnull()
            
            # Get average response and prediction when feature is NULL
            try: rnull = float(pd_data2.groupby("isnull").agg("mean")[response].loc[True])
            except: rnull = 0
            try: rnonzero = float(pd_data2.groupby('isnull').agg('mean')[response].loc[False])
            except: rnonzero = 0
            rzero = 0
            try: pnull = float(pd_data2.groupby("isnull").agg("mean")[prediction].loc[True])
            except: pnull = 0
            try: pnonzero = float(pd_data2.groupby('isnull').agg('mean')[prediction].loc[False])
            except: pnonzero = 0
            pzero = 0
            
            # Get average response for non-NULL feature values, averaged over each histogram bin
            pd_data_nonnull = pd_data2[~fcol.isnull()]
            pd_data_nonnull.reset_index(inplace=True)
            pd_data_nonnull_grp = pd_data_nonnull.groupby(fname).agg("mean")
            
            # Only keep values for the top N categories, and re-order them by exposure
            pd_data_nonnull_grp = pd_data_nonnull_grp[[v in fbin_centres for v in pd_data_nonnull_grp.index.values]]
            ordering = [list(pd_data_nonnull_grp.index.values).index(v) for v in fbin_centres]
            rbins = list(pd_data_nonnull_grp.iloc[ordering][response].values)
            rbin_ids = [0]
            pbins = list(pd_data_nonnull_grp.iloc[ordering][prediction].values)
            pbin_ids = [0]
            fbin_centres = [str(x) for x in fbin_centres]
        
        # Populate dictionaries
        
        # dd is dictionary containing all the stats for one feature, keyed by the stat
        dd["isCat"] = int(fcat)
        dd["min"], dd["max"], dd["mean"], dd["median"], dd["mode"] = float(fmin), float(fmax), float(fmean), float(fmed), str(fmod)
        dd["fracNull"], dd["fracZero"] = float(fnull), float(fzero)
        dd["bins"], dd["counts"] = fbin_centres, [float(x) for x in fcounts]
        
        # d is a dictionary containing everything, keyed by the feature
        d[fname] = dd
        
        # r is a dictionary containing the response trends, keyed by the feature
        # This contains [binIDs, [response for each binID, response non zero, response zero, resposne null]]
        r[fname] = [[float(x) for x in rbin_ids], [float(x) for x in rbins] + [rnonzero, rzero, rnull]]
        
        # p is a dictionary containing the prediction trends, keyed by the feature
        # This contains [binIDs, [prediction for each binID, prediction non zero, prediction zero, prediction null]]
        p[fname] = [[float(x) for x in pbin_ids], [float(x) for x in pbins] + [pnonzero, pzero, pnull]]
    
    return d, r, p


def plot_hists(stats_yml_file_path, response_yml_file_path, prediction_yml_file_path, features, one_ways=None, dashboard=True, normed=False, verbose=True):

    # Load stats
    stats_yaml_file = open(stats_yml_file_path, "r")
    stats = yaml.load(stats_yaml_file)
    stats_yaml_file.close()

    # Close old plots
    plt.close("all")

    # Load the one-ways
    if one_ways != None:
        response_one_way_data = []
        response_yaml_file = open(response_yml_file_path, "r")
        response_one_way_data.append(yaml.load(response_yaml_file))
        response_yaml_file.close()
        
        prediction_one_way_data = []
        prediction_yaml_file = open(prediction_yml_file_path, "r")
        prediction_one_way_data.append(yaml.load(prediction_yaml_file))
        prediction_yaml_file.close()

    # Set line colours
    line_cols = [qc4, qc3, qc2, qc6, qc5, qc7, qc8] # Order of lines: red, blue, orange, green

    # Loop through factors and plot
    allFigs = []
    f = 0
    for fid in range(len(features)):
        facName = features[fid]
        fdesc = facName
        flabel = facName
        if verbose:
            print("Plotting oneways for:", facName)
        fs = stats[facName]

        # Create plots
        fig = plt.figure(f, figsize=(15,8))
        plt.clf()

        # Histogram
        plt.subplot2grid((1,5),(0,0),colspan=4)

        if fs["isCat"] == 0:
            bin_centres = np.asarray(fs["bins"])
            bin_size = np.asarray(fs["bins"][1]) - np.asarray(fs["bins"][0])
            bin_plotsize = 1 * bin_size
            plotX = bin_centres
            plotCount = 100*np.asarray(fs["counts"])/(np.sum(fs["counts"])*1.0)
            plt.bar(plotX, plotCount, width=bin_plotsize, fc=qc1, alpha=0.75, label="Exposure")
            plt.xlabel(flabel)

        else:
            bin_labels, counts = [elem[:20] for elem in fs["bins"]], fs["counts"]
            # Sort them
            #idxSort = np.argsort(counts)[::-1]
            #bin_labels, counts = np.asarray(bin_labels)[idxSort], np.asarray(counts)[idxSort]
            bin_centres, bin_plotsize = np.arange(1,len(counts)+1,1), 1
            plotX = bin_centres
            plotCount = 100*np.asarray(counts)/(np.sum(counts)*1.0)
            plt.bar(plotX, plotCount, width=bin_plotsize, fc=qc1, alpha=0.75, label="Exposure")
            if len(bin_labels) > 5: rot = "vertical"
            else: rot = "horizontal"
            plt.xticks(bin_centres, bin_labels, rotation=rot)
            plt.xlim(bin_centres.min()-(bin_plotsize/2.0), bin_centres.max()+(bin_plotsize/2.0))

        plt.ylabel("Exposure (as % of non-null observations)")

        if one_ways != None:
            # Line plots
            plt.twinx()
            plt.bar(bin_centres[0], 0, width=0, fc=qc1, alpha=0.75, label="Exposure") # not sure what this does....

            # Line plots
            for l in range(1):
                # Load the line dictionary
                # There is one dictionary per line (e.g., response, prediction)
                response_one_way_data_line = response_one_way_data[l]
                prediction_one_way_data_line = prediction_one_way_data[l]

                # Extract the line and convert to numpy array.  The last 3 points are ignored because they are averages over (non-zero, zero, null)
                response_plot_line = response_one_way_data_line[facName][1][0:-3]
                response_plot_line = np.asarray([float(ll) for ll in response_plot_line])
                prediction_plot_line = prediction_one_way_data_line[facName][1][0:-3]
                prediction_plot_line = np.asarray([float(ll) for ll in prediction_plot_line])

                # Normalise lines if required
                if (normed==True):
                    response_plot_line = (response_plot_line - response_plot_line.min())/(response_plot_line.max()-response_plot_line.min())
                    prediction_plot_line = (prediction_plot_line - prediction_plot_line.min())/(prediction_plot_line.max()-prediction_plot_line.min())

                # Some features don't have any exposure at certain bins, so we only plot those with data
                binIDs = (np.asarray(response_one_way_data_line[facName][0])-1).astype(np.int)
                binIDs = (np.asarray(prediction_one_way_data_line[facName][0])-1).astype(np.int)

                # Plot it
                response_legend_label = str('Average Observed').title()
                prediction_legend_label = str('Average Predicted').title()
                if fs["isCat"] == 0:
                    plt.plot(bin_centres[binIDs], response_plot_line, c=qc4, ls='-', lw=3, label=response_legend_label, marker='o', markersize=10, markerfacecolor=qc4, markeredgecolor='k')
                    plt.plot(bin_centres[binIDs], prediction_plot_line, c=qc5, ls='-', lw=3, label=prediction_legend_label, marker='o', markersize=10, markerfacecolor=qc5, markeredgecolor='k')
                else:
                    plt.plot(bin_centres, response_plot_line, c=qc4, ls='-', lw=3, label=response_legend_label, marker='o', markersize=10, markerfacecolor=qc4, markeredgecolor='k')
                    plt.plot(bin_centres, prediction_plot_line, c=qc5, ls='-', lw=3, label=prediction_legend_label, marker='o', markersize=10, markerfacecolor=qc5, markeredgecolor='k')

            if (normed==True):
                plt.ylabel("Average value (normalised)")
            else:
                plt.ylabel("Average value")

            if (normed==True): plt.ylim(-0.01, 1.01)
            plt.gca().get_yaxis().get_major_formatter().set_useOffset(False) # Remove scientific notation from axis

        if fs["isCat"] == 0:
            plt.xlim(plotX.min()-bin_plotsize/2.0,plotX.max()+bin_plotsize/2.0) # This controls xlims of main histogram
            title = facName.upper() + "\n\n" + fdesc + "\n\nMin=" + str(round(fs["min"],2)) + ", Max=" + str(round(fs["max"],2)) + ", Mean=" + str(round(fs["mean"],2)) + ", NULL=" + str(round(100*fs["fracNull"],6)) + "% \n"
        else:
            title = facName.upper() + "\n\n" + fdesc + "\n\nNumber of levels: " + str(int(fs["max"])) + ", NULL=" + str(round(100*fs["fracNull"],6)) + "% \n"

        plt.legend(loc="best")
        plt.title(title)

        # Data/Zeros/NULL dashboard
        plt.subplot2grid((1,5),(0,4))
        if fs["isCat"] == 0:
            plt.bar(1,(1-(fs["fracZero"]+fs["fracNull"])),fc=qc1, width=0.8, alpha=0.75)
            plt.bar(2,fs["fracZero"],fc=qc2, width=0.8)
            plt.bar(3,fs["fracNull"],fc=qc3, width=0.8, alpha=0.75)
            plt.xticks([1,2,3],["Data", "Zero", "NULL"])
            plt.title("% Data/Zero/NULL\n")
        else: 
            plt.bar(1,(1-fs["fracNull"]),fc=qc1, width=0.8, alpha=0.75)
            plt.bar(3,fs["fracNull"],fc=qc3, width=0.8, alpha=0.75)
            plt.xticks([1,3],["Data", "NULL"])
            plt.title("% Data / NULL\n")
        plt.yticks([0,0.25,0.5,0.75,1.0],["0%", "25%", "50%", "75%", "100%"])
        plt.xlim(0.3,3.7)
        plt.ylim(0,1)

        plt.ylabel("Exposure (%)\n")

        if (one_ways != None) and (dashboard==True):
            # Line plots
            plt.twinx()
            plt.bar(bin_centres[0], 0, width=bin_plotsize, fc=qc1, alpha=0.75, label="Exposure") # not sure what this does....

            # Line plots
            for l in range(1):
                # Load the line dictionary
                # There is one dictionary per line (e.g., response, partial dependence, prediction)
                response_one_way_data_line = response_one_way_data[l]
                prediction_one_way_data_line = prediction_one_way_data[l]

                # Extract the (non-zero, zero, null) part
                response_plot_line_dashboard = response_one_way_data_line[facName][1][-3:]
                response_plot_line_dashboard = np.asarray([float(ll) for ll in response_plot_line_dashboard])
                prediction_plot_line_dashboard = prediction_one_way_data_line[facName][1][-3:]
                prediction_plot_line_dashboard = np.asarray([float(ll) for ll in prediction_plot_line_dashboard])

                # Some features don't have any exposure at certain bins, so we only plot those with data
                response_binIDs = (np.asarray(response_one_way_data_line[facName][0])-1).astype(np.int)
                prediction_binIDs = (np.asarray(prediction_one_way_data_line[facName][0])-1).astype(np.int)

                # Plot it
                plt.plot([1,2,3], response_plot_line_dashboard, c=qc4, ls='-', lw=3, marker='o', markersize=10, markerfacecolor=qc4, markeredgecolor='k')
                plt.plot([1,2,3], prediction_plot_line_dashboard, c=qc5, ls='-', lw=3, marker='o', markersize=10, markerfacecolor=qc5, markeredgecolor='k')
            plt.ylabel("Average value")
            plt.gca().get_yaxis().get_major_formatter().set_useOffset(False) # Remove scientific notation from axis

        plt.tight_layout()
        allFigs.append(fig)
        f += 1
    
    return allFigs


#### Adapted from scikitplot.helpers.cumulative_gain_curve and scikitplot.metrics.plot_cumulative_gain
def cumulative_gain_curve(y_true, y_score, pos_label=None):
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    y_true, y_score = np.asarray(y_true), np.asarray(y_score) # not sure why they do this again, but leaving it here as I'm not sure whether removing it will break something

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    
    percentages = np.arange(start=1, stop=len(y_true) + 1)
    percentages = percentages / float(len(y_true))
    percentages = np.insert(percentages, 0, [0])
    percentages = [p*100.0 for p in percentages]
    
    def calculate_gains(y_true, y_score, perfect_model=False):
        
        if perfect_model:
            sorted_indices = np.argsort(y_true)[::-1]
        else:
            sorted_indices = np.argsort(y_score)[::-1]
        
        y_true = y_true[sorted_indices]
        gains = np.cumsum(y_true)

        gains = gains / float(np.sum(y_true))
        gains = np.insert(gains, 0, [0])
        
        return gains
    
    gains = calculate_gains(y_true, y_score, perfect_model=False)
    gains_perfect = calculate_gains(y_true, y_score, perfect_model=True)
    
    return percentages, gains, gains_perfect


def plot_cumulative_gain_single(y_true, y_score, title='Cumulative Gains Curve', ax=None, figsize=None, title_fontsize="large", text_fontsize="medium"):

    # Compute Cumulative Gain Curves
    percentages, gains, gains_perfect = cumulative_gain_curve(y_true, y_score)
    
    percentages_rounded = [round(x, 2) for x in percentages]
    data = pd.DataFrame(np.array([percentages_rounded, gains, gains_perfect]).T, columns=['Percentile', 'Model Gains', 'Perfect Model Gains'])
    gains_data = (
        data[['Percentile', 'Model Gains', 'Perfect Model Gains']]
        .groupby(
            'Percentile',
            as_index=False
        )
        .mean()
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot([0, 100], [0, 1], 'k--', lw=2, label='Random Model')
    ax.plot(gains_data['Percentile'], gains_data['Model Gains'], lw=3, label='Model Gains', c=qc2)
    ax.plot(gains_data['Percentile'], gains_data['Perfect Model Gains'], lw=3, label='Perfect Model Gains', c=qc1)

    ax.set_xlim([0.0, 100.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('Percentile (as ranked by the model)', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='best', fontsize=text_fontsize)

    return fig


def plot_cumulative_gain_multiple(dataset_observed_predictions_dict, response, prediction, title='Cumulative Gains Curves',
                                 ax=None, figsize=None, title_fontsize='large', text_fontsize='medium'):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.set_title(title, fontsize=title_fontsize)
    ax.plot([0, 100], [0, 1], 'k--', lw=2, label='Random Model')
    ax.set_xlim([0.0, 100.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Percentile (as ranked by the model)', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
        
    model_gains_colours = iter([qc2, qc3, qc4, qc5])
    # Compute Cumulative Gain Curves for each set of observations and predictions
    for indx, (dataset, observed_and_predictions_dict) in enumerate(dataset_observed_predictions_dict.items()):
        
        percentages, gains, gains_perfect = cumulative_gain_curve(observed_and_predictions_dict[response], observed_and_predictions_dict[prediction])
        ax.plot(percentages, gains, lw=3, label=f'[{dataset}] Model Gains', c=next(model_gains_colours, qc2))
        
        if indx == 0:
            percentages_rounded = [round(x, 2) for x in percentages]
            data = pd.DataFrame(np.array([percentages_rounded, gains]).T, columns=['Percentile', f'[{dataset}] Model Gains'])
            data_rounded = (
                data[['Percentile', f'[{dataset}] Model Gains']]
                .groupby(
                    'Percentile',
                    as_index=False
                )
                .mean()
            )
            gains_data = data_rounded
            
        elif indx == (len(dataset_observed_predictions_dict.keys()) - 1):
            percentages_rounded = [round(x, 2) for x in percentages]
            data = pd.DataFrame(np.array([percentages_rounded, gains, gains_perfect]).T, columns=['Percentile', f'[{dataset}] Model Gains', 'Perfect Model Gains'])
            data_rounded = (
                data[['Percentile', f'[{dataset}] Model Gains', 'Perfect Model Gains']]
                .groupby(
                    'Percentile',
                    as_index=False
                )
                .mean()
            )
            gains_data = pd.concat([gains_data, data_rounded[[f'[{dataset}] Model Gains', 'Perfect Model Gains']]], axis=1)
            
        else:
            percentages_rounded = [round(x, 2) for x in percentages]
            data = pd.DataFrame(np.array([percentages_rounded, gains]).T, columns=['Percentile', f'[{dataset}] Model Gains'])
            data_rounded = (
                data[['Percentile', f'[{dataset}] Model Gains']]
                .groupby(
                    'Percentile',
                    as_index=False
                )
                .mean()
            )
            gains_data = pd.concat([gains_data, data_rounded[f'[{dataset}] Model Gains']], axis=1)
    
    # Gains_perfect should be the same regardless of split, given a sufficiently large sample size for each split
    # So no need to include in above for loop
    ax.plot(percentages, gains_perfect, lw=3, label='Perfect Model Gains', c=qc1) 
    ax.legend(loc='best', fontsize=text_fontsize)

    return gains_data, fig


def gini(observed, predicted):
    
    # observed and predicted must be numpy arrays    
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)
    
    # check if the data shapes match
    assert observed.shape[0] == predicted.shape[0], 'Gini error: unequal number of rows between observed and prediction'
     
    # concatenate observed, predicted and index 0,1,2,...
    _all = np.asarray(
        np.c_[
            observed,
            predicted,
            np.arange(observed.shape[0])
        ]
        , dtype=np.float
    )
    
    # just set the values of the columns representing each values
    _observed = 0
    _PREDICTED = 1
    _INDEX = 2
    
    # sort by predicted descending, then by index ascending
    sort_order = np.lexsort((_all[:, _INDEX], -1 * _all[:, _PREDICTED]))
    _all = _all[sort_order]
    
    # compute sum of real values for normalisation issues
    total_losses = _all[:, _observed].sum()
    
    # using cumsum on sorted data to measure how 'unsorted' the prediction are ( very hard to read without normalisation)
    gini_sum = _all[:, _observed].cumsum().sum() / total_losses
    
    #centering it to zero
    gini_sum -= (observed.shape[0] + 1.0) / 2.0
    
    # general normalisation
    return gini_sum / observed.shape[0]
 
 
def gini_normalised(observed_and_prediction_data, col_obs='TARGET', col_pred='prediction'):
    
    observed, predicted = observed_and_prediction_data[col_obs].values, observed_and_prediction_data[col_pred].values

    ## normalised gini so that regardless of the data, same ordering returns a value of 1
    return gini(observed, predicted) / gini(observed, observed)


def plot_lift(observed_and_prediction_data, col_obs='TARGET', col_pred='prediction', gini=True, return_data=False):
    
    if gini==True:
        title_gini = " (normalised gini=" + str(round(gini_normalised(observed_and_prediction_data, col_obs=col_obs, col_pred=col_pred),5)) + ")"
    else:
        title_gini = ""
        
    # Sort by predictions
    obs, pred = observed_and_prediction_data[col_obs].values, observed_and_prediction_data[col_pred].values
    sortidx = np.argsort(pred)[::-1]
    obs, pred = obs[sortidx], pred[sortidx]
    l = len(obs)
    
    # Average observed
    avg_o = obs.mean()
    
    # Percentiles
    perc = np.arange(1,101,1)
    lift = []
    cum_lift = []
    prev_n = 0
    for p in perc:
        
        n = int(l * (p/100.)) # how many values up to the pth percentile)
        cum_lift.append(obs[0:n].mean()/avg_o)
        
        lift.append(obs[prev_n:n].mean()/avg_o)
        prev_n = n
    
    
    perc = list(reversed(perc))
        
    # Cumulative lift
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    plt.plot(perc, lift, c=qc3, lw=3, label="Model lift")
    plt.plot(perc, cum_lift, c=qc2, lw=3, label="Cumulative lift")
    ax = plt.gca()
    ax.set_xlim(100, 0)
    plt.legend(loc="best")
    plt.xlabel("Percentile (as ranked by the model)")
    plt.ylabel("Lift")
    plt.title("Lift" + title_gini)
    plt.grid(True)
    plt.tight_layout()

    # Return
    if return_data:
        return fig, lift, cum_lift
    else:
        return fig


def plot_combined_lift(train_validation_lift_data, train_gini='', validation_gini=''):
    
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    plt.plot(train_validation_lift_data['Percentile'], train_validation_lift_data['Train Lift'], c=qc2, lw=3, label="Training Lift")
    plt.plot(train_validation_lift_data['Percentile'], train_validation_lift_data['Validation Lift'], c=qc3, lw=3, label="Validation Lift")
    plt.plot(train_validation_lift_data['Percentile'], train_validation_lift_data['Train Cumulative Lift'], c=qc4, lw=3, label="Training Cumulative Lift")
    plt.plot(train_validation_lift_data['Percentile'], train_validation_lift_data['Validation Cumulative Lift'], c=qc5, lw=3, label="Validation Cumulative Lift")
    ax = plt.gca()
    ax.set_xlim(100, 0)
    plt.legend(loc="best")
    plt.xlabel("Percentile (as ranked by the model)")
    plt.ylabel("Lift")
    plt.title("Lift" + os.linesep + 'Training Gini: ' + str(train_gini) + os.linesep + 'Validation Gini: ' + str(validation_gini))
    plt.grid(True)
    plt.tight_layout()
    
    return fig
