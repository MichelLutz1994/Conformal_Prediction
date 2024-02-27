import numpy as np
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score

from .evaluation import class_wise_performance


def evaluate_mapie(model, X_calib, y_calib, X_new, y_new, alpha=0.1, cv="prefit", method="score", include_last_label=None, classes=None):
    mapie_score = MapieClassifier(model, cv=cv, method=method)
    mapie_score.fit(X_calib, y_calib)
    if include_last_label is not None:
        y_pred, y_set = mapie_score.predict(X_new, alpha=alpha, include_last_label=include_last_label)
    else:
        y_pred, y_set = mapie_score.predict(X_new, alpha=alpha)

    y_set = np.squeeze(y_set)
    cov = classification_coverage_score(y_new, y_set)
    setsize = classification_mean_width_score(y_set)

    if classes is not None:
        classes = classes.classes_
    else:
        classes = list(set(y_calib))

    print("Coverage: {:.2%}".format(cov))
    print("Avg. set size: {:2f}\n".format(setsize))
    print(class_wise_performance(y_new, y_set, classes))
