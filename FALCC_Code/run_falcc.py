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
os.environ["PYTHONWARNINGS"] = "ignore"

#In the following the parameters are specified, see the README in the repository for more information.
#These are just examples for 4 datasets. It works analogously for the other datasets.
input_file_list = ["communities", "implicit30", "social30", "adult_data_set"]
sens_attrs_list = [["race"], ["sensitive"], ["sensitive"], ["sex", "race"]]
favored_list = [(1), (0), (0), (0, 0)]
label_list = ["crime", "label", "label", "salary"]
metric = "demographic_parity"
training = "opt_adaboost"
proxy = "reweigh"
ccr = [-1,-1]
ca = "LOGmeans"
randomstate = -1
#In the datasets of the repository all index columns have been named index
index = "index"
#####################################################################################

for loop, input_file in enumerate(input_file_list):
    sens_attrs = sens_attrs_list[loop]
    label = label_list[loop]
    favored = favored_list[loop]
    allowed = [""]

    link = "Results/" + str(proxy) + "_" + str(input_file) + "/"

    try:
        os.makedirs(link)
    except FileExistsError:
        pass

    randomstate = random.randint(0, 1000)
    #Read the input dataset & split it into training, validation & prediction dataset.
    #Prediction dataset only needed for evaluation.
    df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)

    X = df.loc[:, df.columns != label]
    y = df[label]

    X_train, X_testpred, y_train, y_testpred = train_test_split(X, y, test_size=0.5,
        random_state=100)
    X_test, X_pred, y_test, y_pred = train_test_split(X_testpred, y_testpred,
        test_size=0.3, random_state=randomstate)

    #Run FALCC
    falcc = algorithm.FALCC(link, input_file, df, sens_attrs, favored, label, training, proxy)
    falcc.fit(X_train, y_train, X_test, y_test, metric, weight=0.5)
    df = falcc.predict(X_pred)
    df.to_csv(link + "FALCC_prediction_output.csv", index=False)
