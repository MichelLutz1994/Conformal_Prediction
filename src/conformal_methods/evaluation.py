import pandas as pd
from mapie.metrics import classification_coverage_score, classification_mean_width_score


def class_wise_performance(y_new, y_set, classes):
    df = pd.DataFrame()
    for i in range(len(classes)):
        ynew = y_new.values[y_new.values == i]
        yscore = y_set[y_new.values == i]
        cov = classification_coverage_score(ynew, yscore)
        size = classification_mean_width_score(yscore)

        temp_df = pd.DataFrame({
            "class": [classes[i]],
            "coverage": [cov],
            "avg. set size": [size]
        }, index=[i])

        df = pd.concat([df, temp_df])
    return df


def evaluate_conformal(prediction_set, y_new, classes):
    cov = classification_coverage_score(y_new, prediction_set)
    setsize = classification_mean_width_score(prediction_set)
    print("Coverage: {:.2%}".format(cov))
    print("Avg. set size: {:2f}\n".format(setsize))
    print(class_wise_performance(y_new, prediction_set, classes))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
