"""
Python file/script for evaluation purposes. Runs FALCC only.
"""
import subprocess
import os
import algorithm
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

###We use default argument values for the following parameters, however they can also be set.
###The "check_call" function of the offline phase has to be adapted if they are set.
#testsize = 0.5
#predsize = 0.3
#weight = 0.5
#knn = 15
#list of already trained models (if training is set to "no")
#trained_models_list = [] 

#####################################################################################
#Here we need to specify:
#(1) the dataset name(s)
input_file_list = ["communities", "implicit30", "social30"]
#(2) the name(s) of the sensitive attributes as a list
sens_attrs_list = [["race"], ["sensitive"], ["sensitive"]]
#(3) the value of the favored group
favored_list = [(1), (0), (0)]
#(4) the name of the label
label_list = ["crime", "label", "label"]
#(5) the metric for which the results should be optimized:
#"demographic_parity", "equalized_odds", "equal_opportunity", "treatment_equality"
metric = "demographic_parity"
#(6) which training strategy is used:
#"opt_adaboost" (for our proposed strategy), "opt_random_forest",
#"adaboost" (for our (old) AdaptedAdaboost strategy), "single_classifiers", "no" if own models are used
training = "opt_adaboost"
#(7) if a proxy strategy is used ("no", "reweigh", "remove")
proxy = "reweigh"
#(8) list of allowed "proxy" attributes (required, if reweigh or remove strategy is chosen)
allowed_list = [[""], [""], [""]]
#(9) the minimum and maximum clustersize (if set to -1, we use our automatic approach)
ccr = [-1,-1]
#(10) which cluster parameter estimation strategy to choose (needed depending on ccr)
#"LOGmeans", "elbow"
ca = "LOGmeans"
#(11) randomstate; if set to -1 it will randomly choose a randomstate
randomstate = -1
#(12) run only FALCC or also the other algorithms
testall = False
#(13) if the FairBoost and iFair algorithms should be run
#fairboost_list = [True, True, True]
#ifair_list = [True, True, True]
index = "index"
#####################################################################################

for loop, input_file in enumerate(input_file_list):
    sens_attrs = sens_attrs_list[loop]
    label = label_list[loop]
    favored = favored_list[loop]
    allowed = allowed_list[loop]
    #fairboost = fairboost_list[loop]
    #ifair = ifair_list = [loop]
    
    link = "Results/" + str(proxy) + "_" + str(input_file) + "/"

    try:
        os.makedirs(link)
    except FileExistsError:
        pass

    randomstate = random.randint(0, 1000)
    #Read the input dataset & split it into training, test & prediction dataset.
    #Prediction dataset only needed for evaluation, otherwise size is automatically 0.
    df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
    #Hard set atm

    X = df.loc[:, df.columns != label]
    y = df[label]

    X_train, X_testpred, y_train, y_testpred = train_test_split(X, y, test_size=0.5,
        random_state=randomstate)
    X_test, X_pred, y_test, y_pred = train_test_split(X_testpred, y_testpred,
        test_size=0.3, random_state=randomstate)


    falcc = algorithm.FALCC(link, input_file, df, sens_attrs, favored, label, training, proxy)
    falcc.fit(X_train, y_train, X_test, y_test, metric, weight=0.5)
    df = falcc.predict(X_pred)
    #prediction, model_used, etc.
    df.to_csv(link + "FALCC_prediction_output.csv", index=False)
