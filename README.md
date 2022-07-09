# FALCC

This repository contains the codes needed to reproduce the experiments of our submitted ICDE 2023 paper:
"FALCC: Efficiently performing fair and accurate local classifications through local region clustering"

Run "full_test.py" and use that file to adapt the parameters (e.g. change datasets etc.).
The description is within that file.

The code runs the algorithms FALCC & FALCC-SBT, FALCES and its variants [1], Decouple & Decouple-SBT [2] and FairBoost [3]

[1] Lässig, N., Oppold, S., Herschel, M. "Metrics and Algorithms for Locally Fair and Accurate
    Classifications using Ensembles". 2022.
[2] Dwork, C., Immorlica, N., Kalai, A., Leiserson, M. "Decoupled Classifiers for Group-Fair
    and Efficient Machine Learning". 2018.
[3] Bhaskaruni, D., Hu, H., Lan, C. "Improving Prediction Fairness via Model Ensemble". 2019.

For the algorithms of [2] & [3] we tried to implement the algorithms based on the information provided by the respective papers.


The datasets can be found within 'FALCC_Code/Datasets/'.
The results will be stored within 'FALCC_Code/Results/'.
