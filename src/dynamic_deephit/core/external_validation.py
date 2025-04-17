import numpy as np
import pandas as pd
from dynamic_deephit.core.functions import c_index,weighted_brier_score
from dynamic_deephit.core.main import f_get_risk_predictions
import dynamic_deephit.utils.import_data as impt
def external_validation (model,sess,newdata,df_, id_time_status_list,observation, bin_list, cont_list, pred_time, eval_time,norm_mode='standard',max_length=None):
    """
        Makes predictions using a trained Dynamic-DeepHit model.

        Args:
            model: Trained Dynamic-DeepHit model object
            sess: TensorFlow session used by the model
            newdata: external_validation dataset
            df_:Input DataFrame containing patient data
            id_time_status_list: List of column names for [ID, time, status] columns
            observation(list): observation feature column names.
            bin_list: List of binary feature column names
            cont_list: List of continuous feature column names
            pred_time: List of prediction time points (e.g., [12, 24] months)
            eval_time: List of evaluation windows (e.g., [6, 12] months)
            norm_mode: Normalization mode ('standard' or other)
            max_length : The maximum number of observations for all IDs in the modeling dataset.
Returns
    -------
    Tuple[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]]
        A tuple containing two dictionaries:
        - c_index_mean : Dictionary with same keys as input, where values are
          mean C-index matrices of shape (num_pred_times, num_eval_times)
        - brier_mean : Dictionary with same keys as input, where values are
          mean Brier score matrices of identical shape
    """
    _, DATA, MASK, data_mi, _ = impt.import_dataset(newdata, id_time_status_list, observation,bin_list, cont_list, norm_mode,max_length)
    data, time, label = DATA
    # Prepare training data for Brier score calculation
    _, DATA1, MASK1, data_mi1, _ = impt.import_dataset(df_, id_time_status_list, observation, bin_list,
                                                       cont_list,
                                                       norm_mode, max_length=None)
    tr_data, tr_time, tr_label = DATA1
    risk_all = f_get_risk_predictions(sess, model, data, data_mi, pred_time, eval_time)
    num_Event = newdata[id_time_status_list[2]].nunique() - 1
    num_pred = len(pred_time)
    num_eval = len(eval_time)
    c_index_values = [
        np.zeros((num_pred, num_eval)) for _ in range(num_Event)
    ]
    brier_values = [
        np.zeros((num_pred, num_eval)) for _ in range(num_Event)
    ]
    for p, p_time in enumerate(pred_time):
        for t, t_time in enumerate(eval_time):
            for k in range(num_Event):
                event = k + 1
                # Calculate C-index
                c_idx = c_index(
                    risk_all[k][:, p, t],
                    time,
                    (label[:, 0] == event).astype(int),
                    int(t_time) + int(p_time)
                )

                # Calculate Brier score
                brier = weighted_brier_score(
                    tr_time,
                    (tr_label[:, 0] == event).astype(int),
                    risk_all[k][:, p, t],
                    time,
                    (label[:, 0] == event).astype(int),
                    int(t_time) + int(p_time)
                )
                # Store in matrices
                c_index_values[k][p, t] = c_idx
                brier_values[k][p, t] = brier
    event_names = [f"Event_{i + 1}" for i in range(num_Event)]

    c_index_mean = {
        event_names[i]: pd.DataFrame(
            c_index_values[i],
            index=[f"Pred_{t:.1f}" for t in pred_time],  # 行名：保留1位小数
            columns=[f"Eval_{t:.1f}" for t in eval_time]  # 列名：保留1位小数
        ).rename_axis(
            index="Prediction Time",
            columns="Evaluation Time"
        )
        for i in range(num_Event)
    }
    brier_mean = {
        event_names[i]: pd.DataFrame(
            brier_values[i],
            index=[f"Pred_{t:.1f}" for t in pred_time],
            columns=[f"Eval_{t:.1f}" for t in eval_time]
        ).rename_axis(
            index="Prediction Time",
            columns="Evaluation Time"
        )
        for i in range(num_Event)
    }
    return c_index_mean, brier_mean

