"""
Call methods for the offline phase of the FALCC and FALCES [1] algorithm and its variants.
Also calls the algorithms for FairBoost [2] and Decouple [3]
[1] Lässig, N., Oppold, S., Herschel, M. "Metrics and Algorithms for Locally Fair and Accurate
    Classifications using Ensembles". 2022.
[2] Bhaskaruni, D., Hu, H., Lan, C. "Improving Prediction Fairness via Model Ensemble". 2019.
[3] Dwork, C., Immorlica, N., Kalai, A., Leiserson, M. "Decoupled Classifiers for Group-Fair
    and Efficient Machine Learning". 2018.
"""
import warnings
import argparse
import shelve
import time
import ast
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from kneed import KneeLocator
import algorithm
from algorithm.codes import Metrics
from algorithm.parameter_estimation import log_means

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Directory and name of the input .csv file.")
parser.add_argument("-o", "--output", type=str, help="Directory of the generated output files.")
parser.add_argument("--testsize", default=0.5, type=float, help="Dataset is randomly split into\
    training and test datasets. This value indicates the size of the test dataset. Default value: 0.5")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--sbt", type=str, default=True, help="Choose if you want to split the data\
    before training each classifier. Default: True")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--metric", default="mean", type=str, help="Metric which will be used to test\
    the classifier combinations. Default metric: mean.")
parser.add_argument("--weight", default=0.5, type=float, help="The weight value is used for\
    balancing the importance of accuracy vs fairness. Default value: 0.5")
parser.add_argument("--preprocessed", default=True, type=str, help="Indicates if the dataset\
    has been preprocessed properly. It is a requirement for the cluster variant. Default value: True")
parser.add_argument("--algorithm", default="cluster", type=str, help="Name of the algorithm\
    which should be used. Default: Cluster.")
parser.add_argument("--training", default="adaboost", type=str, help="Name of the model training\
    strategy. Default: adaboost (AdaptedAdaBoost).")
#Some of the following variables are not necessary for all algorithm types.
parser.add_argument("--favored", default=None, type=str, help="Tuple of favored group.\
    Otherwise some metrics can't be used. Default: None.")
parser.add_argument("--threshold", default=0, type=float, help="Gives a threshold for the best\
    global combinations. Needed for the performance-efficient algorithm iff comb-amount is not set.")
parser.add_argument("--comb_amount", default=20, type=int, help="Gives the amount of best global\
    combinations which will be used for the local combination search. This or threshold are needed\
    for performance-efficient algorithm. Default: 20")
parser.add_argument("--knn", default=15, type=int, help="Amount of kNN considered per group for\
    the metrics of the FALCES algorithms without clustering. Default: 15")
parser.add_argument("--falces_version", "--fv", default="NEW", type=str, help="Which FALCES\
    variant (without clustering) is used. The 'NEW' one improves the runtime. Default: NEW")
parser.add_argument("--modelsize", default=6, type=int, help="Number of iterations for the\
    AdaBoost classifier of the clustering algorithm variant. Default: 10")
parser.add_argument("--ccr", default="[-1,-1]", type=str, help="Minimum and maximum amount of clusters\
    that should be generated. If value = -1 the parameter will be estimated. Default: [-1,-1]")
parser.add_argument("--cluster_algorithm", "--ca", default="elbow", type=str, help="Name of the\
    clustering algorithm that should be used to estimate the amount of clusters, if no clustersize\
    is given. Currently supported: [LOGmeans, elbow]. Default: elbow")
parser.add_argument("--fairboost", default=True, type=str, help="Set to true if the FairBoost\
    should and can be run. Default: True")
#Prediction size & randomstate is only important for the evaluation of the model.
parser.add_argument("--predsize", default=0.3, type=float, help="Size of the prediction dataset in\
    relation to the test dataset (only used for the evaluation). Default value: 0.")
parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
args = parser.parse_args()

input_file = args.input
link = args.output
testsize = float(args.testsize)
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
if args.sbt == "True":
    sbt = True
else:
    sbt = False
label = args.label
metric = args.metric
weight = float(args.weight)
if args.preprocessed == "True":
    pre_processed = True
else:
    pre_processed = True
algo_type = args.algorithm
favored = ast.literal_eval(args.favored)
threshold = float(args.threshold)
comb_amount = int(args.comb_amount)
knn_size = int(args.knn)
falces_version = args.falces_version
modelsize = int(args.modelsize)
ccr = ast.literal_eval(args.ccr)
predsize = float(args.predsize)
randomstate = int(args.randomstate)
cluster_algorithm = args.cluster_algorithm
training = args.training
if args.fairboost == "True":
    fairboost = True
else:
    fairboost = False

if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)

#Read the input dataset & split it into training, test & prediction dataset.
#Prediction dataset only needed for evaluation, otherwise size is automatically 0.
df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)

X = df.loc[:, df.columns != label]
y = df[label]

X_train, X_testphases, y_train, y_testphases = train_test_split(X, y, test_size=testsize,
    random_state=randomstate)
X_test, X_pred, y_test, y_pred = train_test_split(X_testphases, y_testphases,
    test_size=predsize, random_state=randomstate)

train_id_list = []
for i, row in X_train.iterrows():
    train_id_list.append(i)

test_id_list = []
for i, row in X_test.iterrows():
    test_id_list.append(i)

pred_id_list = []
for i, row in X_pred.iterrows():
    pred_id_list.append(i)

y_train = y_train.to_frame()
y_test = y_test.to_frame()
y_pred = y_pred.to_frame()


#AdaBoost Training & Testing results of each classifier.
start = time.time()
if training == "single_classifiers":
    model_training_list = ["DecisionTree", "LinearSVM", "NonlinearSVM",\
        "LogisticRegression", "SoftmaxRegression"]
else:
    model_training_list = ["AdaBoost"]

run_main = algorithm.RunFALCES(X_test, y_test, test_id_list, sens_attrs, index, label, link)
if not "sample_weight" in locals():
    sample_weight = None

test_df, d, model_list, model_comb = run_main.train(model_training_list, X_train, y_train,
    sample_weight, modelsize)
test_df.to_csv(link + "testdata_predictions.csv", index_label=index)
test_df = test_df.sort_index()

key_list = []
grouped_df = df.groupby(sens_attrs)
for key, items in grouped_df:
    key_list.append(key)
test_df_sbt, d_sbt, model_list_sbt, model_comb_sbt = run_main.sbt_train(model_training_list,
    X_train, y_train, train_id_list, sample_weight, key_list, modelsize)
test_df_sbt.to_csv(link + "testdata_sbt_predictions.csv", index_label=index)
test_df_sbt = test_df_sbt.sort_index()

#Find the best global model combinations.
#Needed for FALCES-PFA & FALCES-PFA-SBT.
metricer = Metrics(sens_attrs, label)
model_test = metricer.test_score(test_df, model_list)
model_test.to_csv(link + "inaccuracy_testphase.csv", index_label=index)
model_test_sbt = metricer.test_score_sbt(test_df_sbt, d_sbt)
model_test_sbt.to_csv(link + "inaccuracy_testphase_sbt.csv", index_label=index)
model_test_sbt = model_test_sbt.sort_index()

#Single Classifier Predictions
single_class = algorithm.Classifier(index, pred_id_list, sens_attrs, label, model_list)
single_class.single_class_pred(X_pred, y_pred)

#Decouple Algorithm Predictions
decouple_alg = algorithm.Decouple(metricer, index, pred_id_list, sens_attrs, label, favored, model_comb_sbt)
df = decouple_alg.decouple(model_test_sbt, X_pred, y_pred, metric, weight, sbt=True)
df.to_csv(link + "Decouple-SBT_prediction_output.csv", index=False)
decouple_alg = algorithm.Decouple(metricer, index, pred_id_list, sens_attrs, label, favored, model_comb)
df = decouple_alg.decouple(model_test, X_pred, y_pred, metric, weight, sbt=False)
df.to_csv(link + "Decouple_prediction_output.csv", index=False)

#Run the FairBoost Algorithm (if it can run)
if fairboost:
    fb = algorithm.FairBoost(index, pred_id_list, sens_attrs, favored, label, DecisionTreeClassifier())
    df = fb.fit_predict(X_train, y_train, X_pred, y_pred, r=0.1)
    df.to_csv(link + "FairBoost_prediction_output.csv", index=False)

#Estimate the clustersize and then create the clusters
X_test_cluster = copy.deepcopy(X_test)
for sens in sens_attrs:
    X_test_cluster = X_test_cluster.loc[:, X_test_cluster.columns != sens]

#If the clustersize is fixed (hence min and max clustersize has the same value)
if ccr[0] == ccr[1] and ccr[0] != -1:
    clustersize = ccr[0]
else:
    sens_groups = len(X_test.groupby(sens_attrs))
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
    if cluster_algorithm == "LOGmeans":
        clustersize = log_means(X_test_cluster, min_cluster, max_cluster)

    #XMEANS -- Needed XMeans code relies on other implementation and is not published in our repo.
    #if cluster_algorithm == "XMeans":
    #    xm_clust = XMeans(max_cluster)
    #    xm_clust.fit(X_test_cluster.to_numpy())
    #    clustersize = xm_clust.n_clusters

#Save the number of generated clusters as metadata
with open(link + "clustersize.txt", 'w') as outfile:
    outfile.write(str(clustersize))

#Apply the k-means algorithm on the validation dataset
kmeans = KMeans(clustersize).fit(X_test_cluster)
cluster_results = kmeans.predict(X_test_cluster)
X_test_cluster["cluster"] = cluster_results

#Shelve all variables and save it the folder.
filename = link + "cluster.out"
my_shelf = shelve.open(filename, 'n')
for key in dir():
    try:
        my_shelf["kmeans"] = kmeans
    except:
        pass
my_shelf.close()

#Run all offline versions of the FALCC and FALCES algorithms
falcc = algorithm.FALCC(metricer, index, sens_attrs, label, favored, model_list, X_test,
    model_comb, d, pre_processed)
model_dict, kmeans = falcc.cluster_offline(X_test_cluster, kmeans, test_df, metric, weight,
    link, sbt=False)

falccsbt = algorithm.FALCC(metricer, index, sens_attrs, label, favored, model_list_sbt,
    X_test, model_comb_sbt, d_sbt, pre_processed)
model_dict_sbt, kmeans = falccsbt.cluster_offline(X_test_cluster, kmeans, test_df_sbt,
    metric, weight, link, sbt=True)

falces = algorithm.FALCES(metricer, index, sens_attrs, label, favored, model_list,
        X_test, model_comb, d, pre_processed)
global_model_comb = falces.efficient_offline(model_test, metric, weight, threshold, comb_amount)

falcessbt = algorithm.FALCES(metricer, index, sens_attrs, label, favored, model_list_sbt,
        X_test, model_comb_sbt, d_sbt, pre_processed)
global_model_comb_sbt = falcessbt.efficient_offline(model_test_sbt, metric, weight, threshold,
    comb_amount)

falcesnew = algorithm.FALCESNew(metricer, index, sens_attrs, label, favored, model_list,
        X_test, model_comb, d, pre_processed)
falcesnewsbt = algorithm.FALCESNew(metricer, index, sens_attrs, label, favored, model_list_sbt,
        X_test, model_comb_sbt, d_sbt, pre_processed)


#Shelve all variables and save it the folder.
filename = link + "shelve.out"
zmy_shelf = shelve.open(filename, 'n')

for key in dir():
    try:
        zmy_shelf[key] = globals()[key]
    except:
        pass
zmy_shelf.close()
