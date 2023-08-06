"""
This code coordinates the training of each classifier and calls the corresponding training
functions to be executed.
"""
import joblib


class ModelOps():
    """This class calls all classifiers to train their models.

    Parameters
    ----------
    model_dict: dictionary
        Saves information about each trained classifier.

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.
    """
    def __init__(self, model_dict):
        self.model_dict = model_dict


    def return_dict(self):
        """
        Returns the model dictionary.
        """
        return self.model_dict


    def run(self, model_obj, model, folder, input_file, sbt=False, attrs=None):
        """Takes as input the model that will be trained and will return the trained model
        name and will save the model as .pkl & also save some informations in the dictionary.

        Parameter
        -------
        model_obj: Object
            Instance of the Model class.

        model_name: str
            Name of the classifier that should be trained and saved.

        sample_weight: np.array
            Numpy array of the weight of the samples. None, if no reweighing has taken place.

        folder: str
            String of the folder location + prefix.


        Returns
        -------
        joblib_file: str
            Name of the .pkl file of the trained classifier.
        """
        if model == "DecisionTree":
            classifier, prediction, model_name = model_obj.decision_tree()
        elif model == "LinearSVM":
            classifier, prediction, model_name = model_obj.linear_svm()
        elif model == "NonlinearSVM":
            classifier, prediction, model_name = model_obj.nonlinear_svm()
        elif model == "LogisticRegression":
            classifier, prediction, model_name = model_obj.log_regr()
        elif model == "SoftmaxRegression":
            classifier, prediction, model_name = model_obj.softmax_regr()
        elif model == "FaX":
            classifier, prediction, model_name = model_obj.fax()
        elif model == "Fair-SMOTE":
            classifier, prediction, model_name = model_obj.smote()
        elif model == "LFR":
            classifier, prediction, model_name = model_obj.lfr()
        elif model == "AdaBoost":
            classifier_list, prediction_list, model_name = model_obj.adaboost()
            joblist_file_list = []
            with open(folder + 'adaboost.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "RandomForestClassic":
            classifier_list, prediction_list, model_name, rfc = model_obj.rf_classic(n_estimators=attrs[1], max_depth=attrs[2], criterion=attrs[3])
            joblib_file = folder + "RandomForestClassic" + str(attrs[0])  + ".pkl"
            joblib.dump(rfc, joblib_file)
            joblist_file_list = []
            with open(folder + 'rf_classic.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model == "AdaBoostClassic":
            classifier_list, prediction_list, model_name, abc = model_obj.adaboost_classic(n_estimators=attrs[1], max_depth=attrs[2], splitter=attrs[3])
            joblib_file = folder + "AdaBoostClassic" + str(attrs[0])  + ".pkl"
            joblib.dump(abc, joblib_file)
            joblist_file_list = []
            with open(folder + 'adaboost_classic.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list
        elif model in ["OptimizedRandomForest", "OptimizedAdaBoost"]:
            if model == "OptimizedRandomForest":
                classifier_list, prediction_list, model_name = model_obj.opt_learner("RandomForest", input_file, sbt)
                joblist_file_list = []
                with open(folder + 'optimized_random_forest.txt', 'w') as f:
                    f.write(str(classifier_list))
            elif model == "OptimizedAdaBoost":
                classifier_list, prediction_list, model_name = model_obj.opt_learner("AdaBoost", input_file, sbt)
                joblist_file_list = []
                with open(folder + 'optimized_adaboost.txt', 'w') as f:
                    f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list


        d_list = []
        joblib_file = folder + model_name + "_model.pkl"
        joblib.dump(classifier, joblib_file)
        d_list.append(joblib_file)
        d_list.append(prediction)
        #Dictionary containing all models of the following form: {Model Name: [(1) Saved Model
        #as .pkl, (2) Prediction of the model for our test data]
        #Train and save each model on the training data set.
        self.model_dict[joblib_file] = d_list

        return joblib_file
