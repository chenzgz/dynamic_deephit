import numpy as np
from lifelines import KaplanMeierFitter

import random

### C(t)-INDEX CALCULATION
def c_index(Prediction, Time_survival, Death, Time):
    """
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    """
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1

        if (Time_survival[i] <= Time and Death[i] == 1):
            N_t[i, :] = 1

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


### BRIER-SCORE
def brier_score(Prediction, Time_survival, Death, Time):
    #N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)

    return np.mean((Prediction - y_true) ** 2)

    # result2[k, t] = brier_score_loss(risk[:, k], ((te_time[:,0] <= eval_horizon) * (te_label[:,0] == k+1)).astype(int))


##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):
    T = T.reshape([-1])  # (N,) - np array
    Y = Y.reshape([-1])  # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y == 0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  # fill 0 with ZoH (to prevent nan values)

    return G


### C(t)-INDEX CALCULATION
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    """
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    """
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        tmp_idx = np.where(G[0, :] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1. / G[1, -1]) ** 2
        else:
            W = (1. / G[1, tmp_idx[0]]) ** 2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1.  # give weights

        if (T_test[i] <= Time and Y_test[i] == 1):
            N_t[i, :] = 1.

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0, :] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0, :] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
        W[i] = (1. - Y_tilde[i]) * float(Y_test[i]) / G1 + Y_tilde[i] / G2

    y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W * (Y_tilde - (1. - Prediction)) ** 2)




##### USER-DEFINED FUNCTIONS
def f_get_fc_mask1(meas_time, num_Event, num_Category):
    """
        mask1 is required to get the contional probability (to calculate the denominator part)
        mask1 size is [N, num_Event, num_Category]. 1's until the last measurement time
    """
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category])  # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0] + 1)] = 1  # last measurement time

    return mask


def f_get_minibatch(mb_size, x, x_mi, label, time, mask1, mask2, mask3):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb = x[idx, :, :].astype(float)
    x_mi_mb = x_mi[idx, :, :].astype(float)
    k_mb = label[idx, :].astype(float)  # censoring(0)/event(1,2,..) label
    t_mb = time[idx, :].astype(float)
    m1_mb = mask1[idx, :, :].astype(float)  # fc_mask
    m2_mb = mask2[idx, :, :].astype(float)  # fc_mask
    m3_mb = mask3[idx, :].astype(float)  # fc_mask
    return x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb


def f_get_boosted_trainset(x, x_mi, time, label, mask1, mask2, mask3):
    _, num_Event, num_Category = np.shape(mask1)  # dim of mask3: [subj, Num_Event, Num_Category]
    meas_time = np.concatenate([np.zeros([np.shape(x)[0], 1]), np.cumsum(x[:, :, 0], axis=1)[:, :-1]], axis=1)

    total_sample = 0
    for i in range(np.shape(x)[0]):
        total_sample += np.sum(np.sum(x[i], axis=1) != 0)

    new_label = np.zeros([total_sample, np.shape(label)[1]])
    new_time = np.zeros([total_sample, np.shape(time)[1]])
    new_x = np.zeros([total_sample, np.shape(x)[1], np.shape(x)[2]])
    new_x_mi = np.zeros([total_sample, np.shape(x_mi)[1], np.shape(x_mi)[2]])
    new_mask1 = np.zeros([total_sample, np.shape(mask1)[1], np.shape(mask1)[2]])
    new_mask2 = np.zeros([total_sample, np.shape(mask2)[1], np.shape(mask2)[2]])
    new_mask3 = np.zeros([total_sample, np.shape(mask3)[1]])

    tmp_idx = 0
    for i in range(np.shape(x)[0]):
        max_meas = np.sum(np.sum(x[i], axis=1) != 0)

        for t in range(max_meas):
            new_label[tmp_idx + t, 0] = label[i, 0]
            new_time[tmp_idx + t, 0] = time[i, 0]

            new_x[tmp_idx + t, :(t + 1), :] = x[i, :(t + 1), :]
            new_x_mi[tmp_idx + t, :(t + 1), :] = x_mi[i, :(t + 1), :]

            new_mask1[tmp_idx + t, :, :] = f_get_fc_mask1(meas_time[i, t].reshape([-1, 1]), num_Event,
                                                          num_Category)  # age at the measurement
            new_mask2[tmp_idx + t, :, :] = mask2[i, :, :]
            new_mask3[tmp_idx + t, :] = mask3[i, :]

        tmp_idx += max_meas

    return (new_x, new_x_mi, new_time, new_label, new_mask1, new_mask2, new_mask3)

def save_logging(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            f.write('%s:%s\n' % (key, value))


def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here
    return data