import json
from os.path import exists

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from .modelwrapper import ModelWrapper



class GaussianNBModel(ModelWrapper):
    def __init__(self):
        model = make_pipeline(StandardScaler(), GaussianNB())
        super().__init__(model=model, type="GaussianNB")


class SVM(ModelWrapper):
    def __init__(self):
        model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        super().__init__(model=model, type="SVM")


class RandomForest(ModelWrapper):
    def __init__(self):
        model = None
        super().__init__(model=model, type="RandomForest")

    def fit(self, x_train, y_train):
        path = "models/meta_parameters/meta_param_random_forest.json"
        if exists(path):
            print("SVM: load parameter....")
            with open(path, 'r') as f:
                best_params = json.load(f)
        else:
            print("Forest: no parameters available. Start grid search...")
            params = {
                "n_estimators": [10, 50, 100, 500, 1000],
                "max_depth": [None, 1, 2, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            model = RandomForestClassifier()
            random_search = RandomizedSearchCV(
                estimator=model, param_distributions=params, cv=5,
                n_iter=10, random_state=0)
            random_search.fit(x_train, y_train)
            best_params = random_search.best_params_

            with open(path, 'w') as json_file:
                json.dump(best_params, json_file)
        print("Forest: train...")
        self.model = make_pipeline(StandardScaler(),
                                   RandomForestClassifier(
                                       **best_params, random_state=1))
        self.model.fit(x_train, y_train)


class AdaBoost(ModelWrapper):
    def __init__(self):
        model = None
        super().__init__(model=model, type="AdaBoost")

    def fit(self, x_train, y_train):
        path = "models/meta_parameters/meta_param_adaBoost.json"
        if exists(path):
            print("ADA: load parameter....")
            with open(path, 'r') as f:
                best_params = json.load(f)
        else:
            print("Ada: No parameters available. Start grid search...")
            params = {
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 0.2, 0.5, 1],
            }
            model = AdaBoostClassifier()
            random_search = RandomizedSearchCV(
                estimator=model, param_distributions=params, cv=5,
                n_iter=10, random_state=0)
            random_search.fit(x_train, y_train)
            best_params = random_search.best_params_

            with open(path, 'w') as json_file:
                json.dump(best_params, json_file)

        print("ADA: train...")
        self.model = make_pipeline(StandardScaler(), AdaBoostClassifier(
            **best_params, random_state=1))
        self.model.fit(x_train, y_train)