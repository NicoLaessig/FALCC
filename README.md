# FALCC

This repository contains the codes needed to run the FALCC framework of our submitted EDBT 2024 paper:
"FALCC: Efficiently performing locally fair and accurate classifications"

## STRUCTURE

The datasets can be found within `FALCC_Code/Datasets/`.

The results will be stored within `FALCC_Code/Results/`.

In `run_falcc.py` you can see an example on how to call FALCC.


## PARAMETER SETTINGS


**Class initialization**

| Parameter | Default | Values | Definition |
| --- | --- | --- | --- |
| link | - | str | Folder location, where all the files should be saved to. |
| filename | - | str | Name of the dataset. Used when saving the adapted datasets during the proxy discrimination mitigation step. |
| df | - | DataFrame | Full DataFrame. Used for the proxy discrimination mitigation techniques. |
| sens_attrs | - | list of column names | Indicates the protected attributes. |
| favored | - | list of favored values | Indicates the favored value for each protected attribute. Lengths of sens_attrs and favored have to match. |
| label | - | - | Column name of the label. |
| training | "opt_adaboost" | ["opt_adaboost", "opt_random_forest", "adaboost", "single_classifiers", "no"] | Indicates which training strategy should be chosen. The strategy discussed in the FALCC EDBT Paper is "opt_adaboost". Applying hyperparameter tuning using random forest is "opt_random_forest". "adaboost" applies an adapted AdaBoost strategy which varies the classifier type per round (combining heterogeneous and homogeneous ensemble strategies). "single_classifiers" only trains different types of classifiers, like in [1]. "no" indicates that no training strategy is chosen: In this case, already trained classifiers have to be passed via *trained*\_*models*. In preliminary tests, "opt_adaboost" typically returned the most diverse model ensemble. |
| proxy | "reweigh" | ["reweigh", "remove", .] | Name of the proxy discrimination mitigation technique that should be applied. Any other string besides reweigh and remove skips this step. |
| trained_models | None | list of str | List of trained classifiers (by location string) in .pkl format. |
| allowed | "" | list of column names | The attributes that should be ignored when applying the proxy discrimination mitigation strategy. |
| ignore_sens | False | [True, False] | Option to ignore the sensitive attributes when training the classifiers in step 1. |
| sbt | False | [True, False] | Option to train the classifiers separately on each protected group instead of the whole training data. Paper only discusses the False option and only briefly mentions the other. More details within the FALCES framework in [1]. |


**fit method**

The DataFrames available for the training and validation of classifiers should be divided into X_train/y_train and X_test/y_test.
While the same DataFrame can be used both as X_train/y_train and X_test/y_test, it is suggested that they are different subsets of the given DF.

The training data is used to train the single classifiers (step 1 of the FALCC framework). The test data is used for the other steps of the offline phase (clustering and model assessment).
In the experiments we used 50% for the training data (train of classifiers), 35% for the test/validation data (test of trained classifiers) & the final 15% for the prediction data (test of FALCC framework output).

| Parameter | Default | Values | Definition |
| --- | --- | --- | --- |
| metric | "demographic_parity" | ["demographic_parity", "equalized_odds", "equal_opportunity", "treatment_equality"] | Group fairness metric which is used in the loss function. |
| weight | 0.5 | float in [0, 1] | Weight for the loss function: 0 => Only considers fairness, 1 => Only considers accuracy. In experiments of the FALCES framework [1], values in the range [0.3, 0.6] seemed good. |
| cluster_algorithm | "LOGmeans" | ["LOGmeans", "elbow"] | Parameter estimation algorithm used to predict the amount of clusters that should be generated. LOGmeans and Elbow method are the current options, whereas LOGmeans is a lot more efficient [2]. For the elbow method, the elbow point (which originally is something that is manually chosen) is automatically detected using the kneed package. |
| ccr | [-1, -1] | list of two positive integers or -1 | First number indicates the minimum number of clusters generated and the second number indicates the maximum number. Thus, setting it to the same value skips the parameter estimation algorithm step. Default of -1 indicates that the minimum and maximum amount of clusters is chosen automatically (by dataset size). |


**predict method**

Only has the DataFrame X_pred as input that contains the data that should be classified.


**additional information**

For the experiments in the FALCC EDBT Paper we always use the default values given here except for *proxy* and *metric* as mentiond in the Experiments section.


[1] N. Lässig, S. Oppold, M. Herschel. "Metrics and Algorithms for Locally Fair and Accurate Classifications using Ensembles". 2022.

[2] M. Fritz, M. Behringer, H. Schwarz. "LOG-means: Efficiently estimating the number of clusters in large datasets". 2020.
