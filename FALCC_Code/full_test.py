"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import pandas as pd

###We use default argument values for the following parameters, however they can also be set.
###The "check_call" function of the offline phase has to be adapted if they are set.
#testsize = 0.5
#predsize = 0.3
#weight = 0.5
#randomstate = 1
#knn = 15
#cluster_algorithm = "elbow"

#####################################################################################
#Here we need to specify:
#(1) the dataset name,
input_file_list = ["implicit30"]
#(2) the name(s) of the sensitive attributes as a list
sens_attrs_list = [["sensitive"]]
#(3) the value of the favored group
favored_list = [(1)]
#(4) the name of the label
label_list = ["label"]
#(5) the minimum and maximum clustersize (if set to -1, we use our automatic approach)
ccr_list = [[-1,-1]]
#(6) the metric for which the results should be optimized:
#"demographic_parity", "equalized_odds", "equal_opportunity", "treatment_equality"
metric = "demographic_parity"
#(7) if the amount of sensitive groups is binary, the FairBoost algorithm can be run
fairboost_list = [True]
#(8) which training strategy is used:
#"adaboost", "single_classifiers"
training = "adaboost"
#####################################################################################

for loop, input_file in enumerate(input_file_list):
    sensitive = sens_attrs_list[loop]
    label = label_list[loop]
    ccr = ccr_list[loop]
    favored = favored_list[loop]
    fairboost = fairboost_list[loop]

    link = "Results/" + str(input_file) + "/"

    try:
        os.makedirs(link)
    except FileExistsError:
        pass

    #Call the offline and then online phases.
    subprocess.check_call(['python', '-Wignore', 'main_offline.py', '-i', str(input_file),
        '-o', str(link), '--sensitive', str(sensitive), '--label', str(label),
        '--favored', str(favored), '--ccr', str(ccr), '--metric', str(metric),
        '--training', str(training), '--fairboost', str(fairboost)])
    subprocess.check_call(['python', '-Wignore', 'main_online.py', '--folder', str(link)])

    #Call the evaluation
    subprocess.check_call(['python', '-Wignore', 'evaluation.py', '--folder', str(link),
        '--ds', str(input_file), '--sensitive', str(sensitive), '--favored', str(favored),
        '--label', str(label), '--fairboost', str(fairboost)])
