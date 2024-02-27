import numpy as np
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from .evaluation import class_wise_performance


def score_methode(model, X_calib, y_calib, X_new, alpha=0.1):
    n = len(X_calib)
    # 0: get heuristic notion of uncertainty
    raw_score_cal = model.predict_proba(X_calib)
    raw_score_new = model.predict_proba(X_new)
    # 1: get conformal scores.
    cal_scores = 1 - raw_score_cal[np.arange(n), y_calib]
    # 2: get adjusted quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    # 3: form prediction sets
    prediction_sets = raw_score_new >= (1 - qhat)

    return prediction_sets


def adaptive_prediction_methode(model, X_calib, y_calib, X_new, alpha=0.1,
                     lam_reg=0, k_reg = 1, randomisation=True, disallow_zero_sets=False):
    '''

    :param model: pretrained model
    :param X_calib: calibration features
    :param y_calib: calibration labels
    :param X_new: new datapoints to predict
    :param alpha:
    :param lam_reg: for regularisation the higher the greate the punishment
    :param k_reg: bound till more labels will be punished
    :param randomisation: must be set to true to inclulde the last label by change,
            if false the sets become bigger
    :param disallow_zero_sets: set this to False to hold the upper coverage bound,
            set to True so you don't include empty sets
    :return:
    '''
    n = len(X_calib)
    # calculate probabilities
    cal_smx = model.predict_proba(X_calib)

    reg_vec = np.array(k_reg * [0, ] + (cal_smx.shape[1] - k_reg) * [lam_reg, ])[None, :]

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    # index where the true label is
    cal_L = np.where(cal_pi == y_calib.to_numpy()[:, None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
    # Get the score quantile
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')

    # Deploy
    val_smx = model.predict_proba(X_new)

    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec
    indicators = ((val_srt_reg.cumsum(axis=1) - np.random.rand(n_val, 1) * val_srt_reg)
                  <= qhat) if randomisation else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets:
        indicators[:, 0] = True
    prediction_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)
    return prediction_sets