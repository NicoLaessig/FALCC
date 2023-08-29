"""
This file includes the FALCC class & is used to call the 3rd step and online phase
of the FALCC algorithm.
"""
import algorithm
from sklearn.neighbors import NearestNeighbors
import warnings
import argparse
import shelve
import ast
import copy
import math
import joblib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from kneed import KneeLocator
import algorithm
from algorithm.codes import Metrics
from algorithm.parameter_estimation import log_means
#from algorithm.codes.FaX_AI.FaX_methods import MIM
#from algorithm.codes.Fair_SMOTE.SMOTE import smote
#from algorithm.codes.Fair_SMOTE.Generate_Samples import generate_samples
#from algorithm.codes.iFair_helper.iFair import iFair
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.preprocessing import *
from aif360.datasets import BinaryLabelDataset


class FALCC:
    """This class calls runs the 3rd step and online phase of the FALCC algorithm.

    Parameters
    ----------
    link: str
        Link of the output directory.

    filename: str
        Name of the dataset (used for naming).

    df: {array-like, sparse matrix}, shape (n_samples, m_features)
        Whole DataFrame

    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    favored: tuple of float
        Tuple of the values of the favored group.

    label: string
        String name of the target column.

    training: string
        String name of which training procedure is chosen

    proxy: string
        Name of the proxy strategy used

    trained_models: list of strings
        Location of already trained classifiers (they have to be in .pkl format)

    allowed: list of strings
        Feature names that should not be affected by the rpoxy mitigation strategy

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.

    sbt: boolean
        Value is set to true if the classifiers should only be trained on subset of the data
        or on the whole dataset.

    """
    def __init__(self, link, filename, df, sens_attrs, favored, label, training, proxy,
        trained_models=None, allowed="", ignore_sens=False, sbt=False):
        self.link = link
        self.filename = filename
        self.df = df
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.label = label
        self.training = training
        self.proxy = proxy
        self.trained_models = trained_models
        self.allowed = allowed
        self.ignore_sens = ignore_sens
        self.sbt = sbt


    def fit(self, X_train, y_train, X_test, y_test, metric="demographic_parity", weight=0.5,
        cluster_algorithm="LOGmeans", ccr=[-1,-1]):
        """The offline phase of the FALCC algorithm.

        Parameters
        ----------
        X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
            Training data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_train: array-like, shape (n_samples)
            Label vector relative to the training data X_train.

        X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
            Test data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_test: array-like, shape (n_samples)
            Label vector relative to the training data X_test.

        metric: string
            Name of the metric which should be used to get the best result.

        weight: float (0-1)
            Value to balance the accuracy and fairness parts of the metrics.
            Under 0.5: Give fairness higher importance.
            Over 0.5: Give accuracy higher importance.

        cluster_algorithm: string
            String of the parameter estimation algorithm that should be chosen.

        ccr: List of size 2 of integers
            Minimum and maximum amount of clusters that should be generated.
            Unbounded is defined by -1.
        """
        if self.training == "single_classifiers":
            model_training_list = ["DecisionTree", "LinearSVM", "NonlinearSVM",\
                "LogisticRegression", "SoftmaxRegression"]
        #This option requires some additional work to implement FaX, Fair-SMOTE and LFR in FALCC
        elif self.training == "fair":
            model_training_list = ["FaX", "Fair-SMOTE", "LFR"]
        elif self.training == "opt_random_forest":
            model_training_list = ["OptimizedRandomForest"]
        elif self.training == "opt_adaboost":
            model_training_list = ["OptimizedAdaBoost"]

        index = self.df.index.name
        test_id_list = []
        for i, row in X_test.iterrows():
            test_id_list.append(i)
        y_train = y_train.to_frame()
        y_test = y_test.to_frame()

        if self.training != "no":
            run_main = algorithm.RunTraining(X_test, y_test, test_id_list, self.sens_attrs,
                index, self.label, self.favored, self.link, self.ignore_sens)

            test_df, d, model_list, model_comb = run_main.train(model_training_list, X_train, y_train,
                [])
            test_df.to_csv(self.link + "testdata_predictions.csv", index_label=index)
            test_df = test_df.sort_index()

            if self.training != "fair" and self.sbt:
                key_list = []
                grouped_df = df.groupby(self.sens_attrs)
                for key, items in grouped_df:
                    key_list.append(key)
                test_df, d, model_list, model_comb = run_main.sbt_train(model_training_list,
                    X_train, y_train, train_id_list, key_list, [])
                test_df.to_csv(self.link + "testdata_sbt_predictions.csv", index_label=index)
                test_df = test_df.sort_index()
        else:
            d = dict()
            model_list = []
            test_df = pd.DataFrame(columns=[index, self.label])
            test_df[index] = list(y_test.index)
            test_df[label] = y_test[label]
            for sens in self.sens_attrs:
                test_df[sens] = X_test[sens]
            for tm in self.trained_models:
                used_model = joblib.load(tm)
                prediction = used_model.predict(X_test)
                test_df[tm] = prediction
                model_list.append(tm)
                d_list = []
                d_list.append(tm)
                d_list.append(prediction)
                d[tm] = d_list
            groups = len(df.groupby(self.sens_attrs))
            model_comb = list(itertools.combinations_with_replacement(model_list, groups))
            
            test_df.to_csv(self.link + "testdata_predictions.csv", index_label=index)
            test_df = test_df.sort_index()


        #Find the best global model combinations.
        #Needed for FALCES-PFA & FALCES-PFA-SBT -- also for some metadata in FALCC.
        metricer = algorithm.codes.Metrics(self.sens_attrs, self.label)
        model_test = metricer.test_score(test_df, model_list)
        model_test.to_csv(self.link + "inaccuracy_testphase.csv", index_label=index)
        if self.training != "no" and self.training != "fair" and self.sbt:
            model_test = metricer.test_score_sbt(test_df, d)
            model_test.to_csv(self.link + "inaccuracy_testphase_sbt.csv", index_label=index)
            model_test = model_test.sort_index()


        #Estimate the clustersize and then create the clusters
        X_test_new = copy.deepcopy(X_test)
        if self.proxy == "reweigh":
            with open(self.link + "reweighing_attributes.txt", 'w') as outfile:
                df_new = copy.deepcopy(self.df)
                self.weight_dict = dict()
                cols = list(df_new.columns)
                cols.remove(self.label)
                for sens in self.sens_attrs:
                    cols.remove(sens)

                for col in cols:
                    if col in self.allowed:
                        self.weight_dict[col] = 1
                        continue
                    x_arr = df_new[col].to_numpy()
                    col_diff = 0
                    for sens in self.sens_attrs:
                        z_arr = df_new[sens]
                        sens_corr = abs(pearsonr(x_arr, z_arr)[0])
                        if math.isnan(sens_corr):
                            sens_corr = 1
                        col_diff += (1 - sens_corr)
                    col_weight = col_diff/len(self.sens_attrs)
                    self.weight_dict[col] = col_weight
                    df_new[col] *= col_weight
                    X_test_new[col] *= col_weight
                    outfile.write(col + ": " + str(col_weight) + "\n")
            df_new.to_csv("Datasets/reweigh/" + self.filename + ".csv", index_label=index)
        elif self.proxy == "remove":
            with open(self.link + "removed_attributes.txt", 'w') as outfile:
                df_new = copy.deepcopy(self.df)
                self.weight_dict = dict()
                cols = list(df_new.columns)
                cols.remove(self.label)
                for sens in self.sens_attrs:
                    cols.remove(sens)

                for col in cols:
                    cont = False
                    if col in self.allowed:
                        self.weight_dict[col] = 1
                        continue
                    x_arr = df_new[col].to_numpy()
                    col_diff = 0
                    for sens in self.sens_attrs:
                        z_arr = df_new[sens]
                        pearson = pearsonr(x_arr, z_arr)
                        sens_corr = abs(pearson[0])
                        if math.isnan(sens_corr):
                            sens_corr = 1
                        if sens_corr > 0.5 and pearson[1] < 0.05:
                            X_test_new = X_test_new.loc[:, X_test_new.columns != col]
                            cont = True
                            outfile.write(col + "\n")
                            continue
                    if not cont:
                        self.weight_dict[col] = 1
                df_new.to_csv("Datasets/removed/" + self.filename + ".csv", index_label=index)

        X_test_cluster = copy.deepcopy(X_test_new)
        for sens in self.sens_attrs:
            X_test_cluster = X_test_cluster.loc[:, X_test_cluster.columns != sens]

        #If the clustersize is fixed (hence min and max clustersize has the same value)
        if ccr[0] == ccr[1] and ccr[0] != -1:
            clustersize = ccr[0]
        else:
            sens_groups = len(X_test_new.groupby(self.sens_attrs))
            if ccr[0] == -1:
                min_cluster = min(len(X_test_cluster.columns), int(len(X_test_cluster)/(50*sens_groups)))
            else:
                min_cluster = ccr[0]
            if ccr[1] == -1:
                max_cluster = min(int(len(X_test_cluster.columns)**2/2), int(len(X_test_cluster)/(10*sens_groups)))
            else:
                max_cluster = ccr[1]

            #ELBOW
            #The following implements the Elbow Method, using the KneeLocator to perform the
            #manual step of finding the elbow point.
            if cluster_algorithm == "elbow":
                k_range = range(min_cluster, max_cluster)
                inertias = []
                for k in k_range:
                    km = KMeans(n_clusters = k)
                    km.fit(X_test_cluster)
                    inertias.append(km.inertia_)
                y = np.zeros(len(inertias))

                kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
                clustersize = kn.knee - 1

            #LOGMEANS
            #Calls the LOGMeans method instead as the parameter estimation algorithm.
            if cluster_algorithm == "LOGmeans":
                clustersize = log_means(X_test_cluster, min_cluster, max_cluster)

        #Save the number of generated clusters as metadata
        with open(self.link + "clustersize.txt", 'w') as outfile:
            outfile.write(str(clustersize))

        #Apply the k-means algorithm on the validation dataset
        self.kmeans = KMeans(clustersize).fit(X_test_cluster)
        cluster_results = self.kmeans.predict(X_test_cluster)
        X_test_cluster["cluster"] = cluster_results

        #Shelve all variables and save it the folder.
        """
        filename = self.link + "cluster.out"
        my_shelf = shelve.open(filename, 'n')
        for key in dir():
            try:
                my_shelf["kmeans"] = self.kmeans
            except:
                pass
        my_shelf.close()
        """

        if self.proxy == "no":
            self.weight_dict = None

        clustered_df = X_test_cluster.groupby("cluster")
        self.model_dict = dict()
        column_list = test_df.columns

        groups = test_df[self.sens_attrs].drop_duplicates(self.sens_attrs).reset_index(drop=True)
        actual_num_of_groups = len(groups)
        sensitive_groups = []
        sens_cols = groups.columns
        for i, row in groups.iterrows():
            sens_grp = []
            for col in sens_cols:
                sens_grp.append(row[col])
            sensitive_groups.append(tuple(sens_grp))

        for key, item in clustered_df:
            part_df = clustered_df.get_group(key)
            part_df2 = test_df.merge(part_df, on="index", how="inner")[column_list]
            groups2 = part_df2[self.sens_attrs].drop_duplicates(self.sens_attrs).reset_index(drop=True)
            num_of_groups = len(groups2)
            cluster_sensitive_groups = []
            for i, row in groups2.iterrows():
                sens_grp = []
                for col in sens_cols:
                    sens_grp.append(row[col])
                cluster_sensitive_groups.append(tuple(sens_grp))

            #If a cluster does not contain samples of all groups, it will take the k nearest neighbors
            #(default value = 15) to test the model combinations
            if num_of_groups != actual_num_of_groups:
                cluster_center = self.kmeans.cluster_centers_[key]
                for sens_grp in sensitive_groups:
                    if sens_grp not in cluster_sensitive_groups:
                        if len(self.sens_attrs) == 1:
                            sens_grp = sens_grp[0]
                        grouped_df = X_test.groupby(self.sens_attrs)
                        for key2, item2 in grouped_df:
                            if key2 == sens_grp:
                                knn_df = grouped_df.get_group(key2)
                                for sens_attr in self.sens_attrs:
                                    knn_df = knn_df.loc[:, knn_df.columns != sens_attr]
                                nbrs = NearestNeighbors(n_neighbors=15, algorithm='kd_tree').fit(knn_df.values)
                                print(cluster_center.shape)
                                print(knn_df.shape)
                                indices = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=False)
                                real_indices = X_test.index[indices].tolist()
                                nearest_neighbors_df = test_df.loc[real_indices[0]]
                                part_df2 = part_df2.append(nearest_neighbors_df)

            if not self.sbt:
                model_test = metricer.test_score(part_df2, model_list)
                model_test.to_csv(self.link + str(key) + "_inaccuracy_testphase.csv",
                    index_label=index)
            else:
                model_test = metricer.test_score_sbt(part_df2, d)
                model_test.to_csv(self.link + str(key) + "_inaccuracy_testphase_sbt.csv",
                    index_label=index)
            comb_list_global, group_tuple = metricer.fairness_metric(model_test,
                model_comb, self.favored, metric, weight, comb_amount=1)

            subdict = dict()
            for i, gt in enumerate(group_tuple):
                dict_key = []
                for j in gt:
                    dict_key.append(float(j))
                subdict[str(dict_key)] = comb_list_global[0][i]

            self.model_dict[key] = subdict



    def predict(self, X_pred):
        """This function predicts the label of each prediction sample for FALCC/FALCC-SBT
        (the online phase).

        Parameters
        ----------
        X_pred: {array-like, sparse matrix}, shape (n_samples, m_features)
            Prediction data vector, where n_samples is the number of samples and
            m_features is the number of features.


        Returns/Output
        ----------
        pred_df: Output DataFrame
            Contains: index, value of sensitive attributes, label, predicted value,
            model used for prediction, model combination used for prediction.
        """
        if not self.sbt:
            cluster_model = "FALCC"
        else:
            cluster_model = "FALCC-SBT"

        index = self.df.index.name
        pred_df = pd.DataFrame(columns=[index, cluster_model, "model_used"])
        pred_count = 0


        sens_count = 1
        X_pred_cluster = copy.deepcopy(X_pred)
        for attr in self.sens_attrs:
            pred_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1
            X_pred_cluster = X_pred_cluster.loc[:, X_pred_cluster.columns != attr]

        if self.proxy in ("reweigh", "remove"):
            for col in list(X_pred_cluster.columns):
                if col in self.weight_dict:
                    X_pred_cluster[col] *= self.weight_dict[col]
                else:
                    X_pred_cluster = X_pred_cluster.loc[:, X_pred_cluster.columns != col]

        Z_pred = copy.deepcopy(X_pred)
        if self.ignore_sens:
            for sens in self.sens_attrs:
                Z_pred = Z_pred.loc[:, Z_pred.columns != sens]
        Z2_pred = copy.deepcopy(Z_pred)

        if self.training == "fair":
            ##For FaX if needed
            X3 = copy.deepcopy(X_pred)
            X3 = X3.loc[:, X3.columns != self.sens_attrs[0]]

            ##For LFR if needed
            lfr_pred_df = pd.merge(X_pred, y_pred, left_index=True, right_index=True)
            dataset_pred = BinaryLabelDataset(df=lfr_pred_df, label_names=[self.label], protected_attribute_names=self.sens_attrs)
            lfr_model = joblib.load(self.link + "LFR_model.pkl")
            dataset_transf_pred = lfr_model.transform(dataset_pred)
            lfr_preds = list(dataset_transf_pred.labels)
            lfr_prediction = [lfr_preds[i][0] for i in range(len(lfr_preds))]

        for i in range(len(X_pred)):
            sens_value = []
            for attr in self.sens_attrs:
                sens_value.append(float(X_pred.iloc[i][attr]))

            cluster_results = self.kmeans.predict(X_pred_cluster.iloc[i].values.reshape(1, -1))

            model = self.model_dict[cluster_results[0]][str(sens_value)]
            used_model = joblib.load(model)
            if str(sens_value) == "[0.0]":
                Z2_pred.iloc[i][self.sens_attrs[0]] = 1.0
            elif str(sens_value) == "[1.0]":
                Z2_pred.iloc[i][self.sens_attrs[0]] = 0.0
            elif str(sens_value) == "[1]":
                Z2_pred.iloc[i][self.sens_attrs[0]] = 0
            else:
                Z2_pred.iloc[i][self.sens_attrs[0]] = 1

            if "FaX" in model:
                prediction = used_model.predict(X3.iloc[i].values.reshape(1, -1))[0]
            elif "LFR" in model:
                prediction = lfr_prediction[i]
            else:
                prediction = used_model.predict(Z_pred.iloc[i].values.reshape(1, -1))[0]


            pred_df.at[pred_count, index] = X_pred.index[i]
            for attr in self.sens_attrs:
                pred_df.at[pred_count, attr] = X_pred.iloc[i][attr]
            #pred_df.at[pred_count, self.label] = y_pred.iloc[i].values[0]
            pred_df.at[pred_count, cluster_model] = prediction
            pred_df.at[pred_count, "model_used"] = model

            pred_count = pred_count + 1

        return pred_df
