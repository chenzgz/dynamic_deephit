import numpy as np
import pandas as pd
from dynamic_deephit.core.main import train_dynamic_deephit_model,f_get_risk_predictions
import dynamic_deephit.utils.import_data as impt


# noinspection PyPackageRequirements
def predict_with_dynamic_deephit(model, sess, df_, id_time_status_list,observation, bin_list, cont_list, pred_time, eval_time,
                                 norm_mode='standard',max_length=None):
    """
    Makes predictions using a trained Dynamic-DeepHit model.

    Args:
        model: Trained Dynamic-DeepHit model object
        sess: TensorFlow session used by the model
        df_: Input DataFrame containing patient data
        id_time_status_list: List of column names for [ID, time, status] columns
        observation(list): observation feature column names.
        bin_list: List of binary feature column names
        cont_list: List of continuous feature column names
        pred_time: List of prediction time points (e.g., [12, 24] months)
        eval_time: List of evaluation windows (e.g., [6, 12] months)
        norm_mode: Normalization mode ('standard' or other)
        max_length : The maximum number of observations for all IDs in the modeling dataset.

    Returns:
        Dictionary containing:
        - risk_predictions: Dictionary of risk scores per event type
        - survival_probabilities: Array of survival probabilities
        - cumulative_incidence: Dictionary of cumulative incidence functions
    """
    id_list = pd.unique(df_[id_time_status_list[0]])
    # Data preprocessing (matches training pipeline)
    # Returns:
    # - DATA: Tuple of (processed_data, time, label)
    # - MASK: Tuple of masks
    # - data_mi: Missing data indicators
    _, DATA, MASK, data_mi, _ = impt.import_dataset(df_, id_time_status_list, observation,bin_list, cont_list, norm_mode,max_length)
    data, _, _ = DATA  # Extract the processed feature data

    # Core prediction - get risk scores for all patients, time points and events
    # Returns dictionary where keys are event types and values are 3D arrays:
    # [patients, prediction_times, evaluation_windows]
    risk_all = f_get_risk_predictions(sess, model, data, data_mi, pred_time, eval_time)

    # Get dimensions from the data
    num_patients = data.shape[0]  # Number of patients in input
    num_events = len(risk_all)  # Number of competing events

    # Initialize output arrays:
    # survival_probs: 3D array [patients, pred_times, eval_windows]
    # Starts with 1.0 (100% survival) for all
    survival_probs = np.ones((num_patients, len(pred_time), len(eval_time)))

    # cumulative_incidence: Dict of 3D arrays (same shape as survival_probs)
    # One array per event type, initialized to zeros
    cumulative_incidence = {k: np.zeros_like(survival_probs) for k in range(num_events)}

    # Populate results from risk predictions
    for k in risk_all.keys():  # For each event type
        # Cumulative incidence equals the predicted risk scores
        cumulative_incidence[k] = risk_all[k]

        # Update survival probabilities:
        # Multiply existing survival prob by (1 - current event risk)
        # This implements the complement rule for competing risks
        for p in range(risk_all[k].shape[1]):  # For each prediction time
            for t in range(risk_all[k].shape[2]):  # For each evaluation window
                survival_probs[:, p, t] *= (1 - risk_all[k][:, p, t])

    event_names = [f"Event_{i + 1}" for i in range(num_events)]
    ID_name=[f"ID_{id_list[j]}" for j in range(num_patients)]
    cumulative_incidence = {
        event_names[i]:{ID_name[j]:pd.DataFrame(
            cumulative_incidence[i][j,:],
            index=[f"Pred_{t:.1f}" for t in pred_time],
            columns=[f"Eval_{t:.1f}" for t in eval_time]
        ).rename_axis(
            index="Prediction Time",
            columns="Evaluation Time"
        )   for j in range(num_patients)}
        for i in range(num_events)
    }
    survival_probs = {
        ID_name[j]: pd.DataFrame(
            survival_probs[j, :, :],
            index=[f"Pred_{t:.1f}" for t in pred_time],
            columns=[f"Eval_{t:.1f}" for t in eval_time]
        )
        for j in range(num_patients)
    }
    return {'cumulative_incidence':cumulative_incidence,
    'survival_probs':survival_probs}
