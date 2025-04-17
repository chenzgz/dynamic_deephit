from sklearn.model_selection import KFold
import numpy as np
from dynamic_deephit.core.main import train_dynamic_deephit_model,f_get_risk_predictions
import dynamic_deephit.utils.import_data as impt
from dynamic_deephit.core.functions import c_index,weighted_brier_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf
import pandas as pd

def interval_CV(method,df_, id_time_status_list,observation, bin_list, cont_list, pred_time, eval_time, file_path, hyperparams,
                            K=5,n=1,train_rate=0.8, seed=42, norm_mode='standard', burn_in_mode='ON', boost_mode='ON',max_length=None):
    """
        Perform interval cross-validation for the Dynamic-DeepHit model with evaluation for each pred_time and eval_time combination.

        Args:
            df_ (pd.DataFrame): The input dataset.
            id_time_status_list (list): List of column names for ID, time, and status.
            observation(list): observation feature column names.
            bin_list (list): List of binary feature column names.
            cont_list (list): List of continuous feature column names.
            pred_time (list): List of prediction time points.
            eval_time (list): List of evaluation time points.
            file_path (str): Path to save the trained models.
            hyperparams (dict): Dictionary of hyperparameters.
            k (int, optional): The specific number of folds in the K-fold cross-validation method. Defaults to 5.
            n (int): internal cross-validation. For random splitting, it refers to the number of random splitting iterations; for K-fold cross-validation, it refers to the number of outer loop iterations; for bootstrap, it refers to the number of resampling iterations. Defaults to 1.
            train_rate(float, optional): The proportion for random splitting. Defaults to 0.8.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            norm_mode (str, optional): Normalization mode for continuous features. Defaults to 'standard'.
            burn_in_mode (str, optional): Whether to perform burn-in training. Defaults to 'ON'.
            boost_mode (str, optional): Whether to boost the training set. Defaults to 'ON'.
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
    if method=="K-fold":
        c_index_mean, brier_mean=k_fold_cross_validation(df_, id_time_status_list,observation, bin_list, cont_list, pred_time, eval_time, file_path, hyperparams,
                            K,n, seed, norm_mode, burn_in_mode, boost_mode,max_length)
    elif method=="split":
        c_index_mean, brier_mean =train_test_split_validation(
        df_, id_time_status_list, observation,bin_list, cont_list, pred_time, eval_time,
        file_path, hyperparams,n, train_rate, seed, norm_mode,
        burn_in_mode, boost_mode,max_length)
    elif method=="boot":
        c_index_mean, brier_mean =bootstrap_validation(
        df_, id_time_status_list,observation, bin_list, cont_list, pred_time, eval_time,
        file_path, hyperparams, n, seed, norm_mode,burn_in_mode, boost_mode,max_length)
    else:
        raise ValueError(f"Unsupported validation method: {method}. Choose from 'K-fold', 'split' or 'boot'")
    return c_index_mean, brier_mean



def k_fold_cross_validation(df_, id_time_status_list,observation, bin_list, cont_list, pred_time, eval_time, file_path, hyperparams,
                            K=5,n=1, seed=42, norm_mode='standard', burn_in_mode='ON', boost_mode='ON',max_length=None):
    """
    Perform K-fold cross-validation for the Dynamic-DeepHit model with evaluation for each pred_time and eval_time combination.

    Args:
        df_ (pd.DataFrame): The input dataset.
        id_time_status_list (list): List of column names for ID, time, and status.
        observation(list): observation feature column names.
        bin_list (list): List of binary feature column names.
        cont_list (list): List of continuous feature column names.
        pred_time (list): List of prediction time points.
        eval_time (list): List of evaluation time points.
        file_path (str): Path to save the trained models.
        hyperparams (dict): Dictionary of hyperparameters.
        k (int, optional): Number of folds. Defaults to 5.
        n (int): Number of repetitions for K-fold cross validation. Defaults to 1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        norm_mode (str, optional): Normalization mode for continuous features. Defaults to 'standard'.
        burn_in_mode (str, optional): Whether to perform burn-in training. Defaults to 'ON'.
        boost_mode (str, optional): Whether to boost the training set. Defaults to 'ON'.
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
    all_c_index = []
    all_brier = []
    for i in range(n):
        try:
            # Initialize KFold
            kf = KFold(n_splits=K, shuffle=True, random_state=seed + i)

            num_Event = df_[id_time_status_list[2]].nunique() - 1
            num_pred = len(pred_time)
            num_eval = len(eval_time)

            # Initialize result matrices
            c_index_matrices = [
                np.zeros((K, num_pred, num_eval)) for _ in range(num_Event)
            ]
            brier_matrices = [
                np.zeros((K, num_pred, num_eval)) for _ in range(num_Event)
            ]
            # Perform K-fold cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(df_)):
                print(f"Training Fold {fold + 1}/{K}")

                # Split data into training and validation sets
                train_df = df_.iloc[train_idx]
                val_df = df_.iloc[val_idx]

                # Train the model on the training set
                model, sess = train_dynamic_deephit_model(
                    train_df, id_time_status_list, observation, bin_list, cont_list, pred_time, eval_time, file_path,
                    hyperparams,
                    seed=seed, norm_mode=norm_mode, burn_in_mode=burn_in_mode, boost_mode=boost_mode, max_length=None
                )
                graph = tf.get_default_graph()
                input_tensor = graph.get_tensor_by_name("Dynamic-DeepHit/Placeholder_5:0")
                max_length = input_tensor.shape[1].value

                # Prepare training data for Brier score calculation
                _, DATA1, MASK1, data_mi1, _ = impt.import_dataset(train_df, id_time_status_list, observation, bin_list,
                                                                   cont_list,
                                                                   norm_mode, max_length=None)
                tr_data, tr_time, tr_label = DATA1

                # Prepare validation data
                _, DATA, MASK, data_mi, _ = impt.import_dataset(val_df, id_time_status_list, observation, bin_list,
                                                                cont_list, max_length=max_length)
                va_data, va_time, va_label = DATA

                # Get risk predictions
                risk_all = f_get_risk_predictions(sess, model, va_data, data_mi, pred_time, eval_time)

                # Calculate metrics for each combination
                for p, p_time in enumerate(pred_time):

                    for t, t_time in enumerate(eval_time):
                        for k in range(num_Event):
                            event = k + 1
                            # Calculate C-index
                            c_idx = c_index(
                                risk_all[k][:, p, t],
                                va_time,
                                (va_label[:, 0] == event).astype(int),
                                int(t_time) + int(p_time)
                            )

                            # Calculate Brier score
                            brier = weighted_brier_score(
                                tr_time,
                                (tr_label[:, 0] == event).astype(int),
                                risk_all[k][:, p, t],
                                va_time,
                                (va_label[:, 0] == event).astype(int),
                                int(t_time) + int(p_time)
                            )
                            # Store in matrices
                            c_index_matrices[k][fold, p, t] = c_idx
                            brier_matrices[k][fold, p, t] = brier
                print(f"Completed evaluation for Fold {fold + 1}")

            avg_c_index = [np.zeros_like(mat[0]) for mat in c_index_matrices]
            avg_brier = [np.zeros_like(mat[0]) for mat in brier_matrices]
            for event_idx in range(num_Event):
                avg_c_index[event_idx] = np.nanmean(c_index_matrices[event_idx], axis=0)
                avg_brier[event_idx] = np.nanmean(brier_matrices[event_idx], axis=0)
            all_c_index.append(avg_c_index)
            all_brier.append(avg_brier)
        except Exception as e:
            print(f"Error in iteration i={i}: {e}")
            continue
    mean_c_index = [
        np.nanmean([result[event_idx] for result in all_c_index], axis=0)
        for event_idx in range(num_Event)
    ]

    mean_brier = [
        np.nanmean([result[event_idx] for result in all_brier], axis=0)
        for event_idx in range(num_Event)
    ]
    event_names = [f"Event_{i + 1}" for i in range(num_Event)]

    c_index_mean= {
        event_names[i]: pd.DataFrame(
            mean_c_index[i],
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
            mean_brier[i],
            index=[f"Pred_{t:.1f}" for t in pred_time],
            columns=[f"Eval_{t:.1f}" for t in eval_time]
        ).rename_axis(
            index="Prediction Time",
            columns="Evaluation Time"
        )
        for i in range(num_Event)
    }


    return  c_index_mean,brier_mean


def train_test_split_validation(
        df_, id_time_status_list, observation,bin_list, cont_list, pred_time, eval_time,
        file_path, hyperparams,n, train_rate=0.8, seed=42, norm_mode='standard',
        burn_in_mode='ON', boost_mode='ON',max_length=None):
    """
    Perform train-test split validation for the Dynamic-DeepHit model with evaluation
    for each pred_time and eval_time combination.

    Args:
        df_ (pd.DataFrame): Input dataset.
        id_time_status_list (list): Columns for ID, time, and status.
        observation(list): observation feature column names.
        bin_list (list): Binary feature columns.
        cont_list (list): Continuous feature columns.
        pred_time (list): Time points for prediction.
        eval_time (list): Time points for evaluation.
        file_path (str): Path to save the model.
        hyperparams (dict): Model hyperparameters.
        n: Number of repetitions for split dataset.
        train_rate (float, optional): Proportion of training data (default: 0.8).
        seed (int, optional): Random seed (default: 42).
        norm_mode (str, optional): Normalization mode (default: 'standard').
        burn_in_mode (str, optional): Whether to use burn-in training (default: 'ON').
        boost_mode (str, optional): Whether to boost the training set (default: 'ON').
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
    all_c_index = []
    all_brier = []
    for i in range(n):
        try:
            # Split data into training and validation sets
            train_df, val_df = train_test_split(
                df_, train_size=train_rate, random_state=seed + i
            )

            # Prepare training data for Brier score calculation
            _, (tr_data, tr_time, tr_label), _, _, _ = impt.import_dataset(
                train_df, id_time_status_list, observation, bin_list, cont_list, norm_mode
            )

            # Train the model on the training set
            model, sess = train_dynamic_deephit_model(
                train_df, id_time_status_list, observation, bin_list, cont_list,
                pred_time, eval_time, file_path, hyperparams,
                seed=seed, norm_mode=norm_mode,
                burn_in_mode=burn_in_mode, boost_mode=boost_mode
            )
            graph = tf.get_default_graph()
            input_tensor = graph.get_tensor_by_name("Dynamic-DeepHit/Placeholder_5:0")
            max_length = input_tensor.shape[1].value
            # 4. Prepare validation data
            _, (va_data, va_time, va_label), _, va_data_mi, _ = impt.import_dataset(
                val_df, id_time_status_list, observation, bin_list, cont_list, norm_mode, max_length=max_length
            )

            # Get risk predictions
            risk_all = f_get_risk_predictions(sess, model, va_data, va_data_mi, pred_time, eval_time)

            # Calculate metrics for each combination
            num_Event = df_[id_time_status_list[2]].nunique() - 1
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
                            va_time,
                            (va_label[:, 0] == event).astype(int),
                            int(t_time) + int(p_time)
                        )

                        # Calculate Brier score
                        brier = weighted_brier_score(
                            tr_time,
                            (tr_label[:, 0] == event).astype(int),
                            risk_all[k][:, p, t],
                            va_time,
                            (va_label[:, 0] == event).astype(int),
                            int(t_time) + int(p_time)
                        )
                        # Store in matrices
                        c_index_values[k][p, t] = c_idx
                        brier_values[k][p, t] = brier
            all_c_index.append(c_index_values)
            all_brier.append(brier_values)
        except Exception as e:
            print(f"Error in iteration i={i}: {e}")
            continue
    mean_c_index = [
        np.nanmean([result[event_idx] for result in all_c_index], axis=0)
        for event_idx in range(num_Event)
    ]

    mean_brier = [
        np.nanmean([result[event_idx] for result in all_brier], axis=0)
        for event_idx in range(num_Event)
    ]
    event_names = [f"Event_{i + 1}" for i in range(num_Event)]

    c_index_mean = {
        event_names[i]: pd.DataFrame(
            mean_c_index[i],
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
            mean_brier[i],
            index=[f"Pred_{t:.1f}" for t in pred_time],
            columns=[f"Eval_{t:.1f}" for t in eval_time]
        ).rename_axis(
            index="Prediction Time",
            columns="Evaluation Time"
        )
        for i in range(num_Event)
    }

    return c_index_mean,brier_mean


def bootstrap_validation(
        df_, id_time_status_list,observation, bin_list, cont_list, pred_time, eval_time,
        file_path, hyperparams, n_bootstrap=200, seed=42, norm_mode='standard',
        burn_in_mode='ON', boost_mode='ON',max_length=None):
    """
    Perform bootstrap validation for the Dynamic-DeepHit model with evaluation
    for each pred_time and eval_time combination.

    Args:
        df_ (pd.DataFrame): Input dataset.
        id_time_status_list (list): Columns for ID, time, and status.
        observation(list): observation feature column names.
        bin_list (list): Binary feature columns.
        cont_list (list): Continuous feature columns.
        pred_time (list): Time points for prediction.
        eval_time (list): Time points for evaluation.
        file_path (str): Path to save the model.
        hyperparams (dict): Model hyperparameters.
        n_bootstrap (int, optional): Number of bootstrap samples (default: 200).
        seed (int, optional): Random seed (default: 42).
        norm_mode (str, optional): Normalization mode (default: 'standard').
        burn_in_mode (str, optional): Whether to use burn-in training (default: 'ON').
        boost_mode (str, optional): Whether to boost the training set (default: 'ON').
        max_length : The maximum number of observations for all IDs in the modeling dataset.

    Returns:
        dict: Results including:
            - apparent_performance: Apparent performance on original data
            - optimism_corrected: Optimism-corrected performance
            - bootstrap_results: All bootstrap results
            - avg_optimism: Average optimism for each metric
    """
    np.random.seed(seed)


    num_Event = df_[id_time_status_list[2]].nunique() - 1
    #  Train model on original data and get apparent performance
    print("Calculating apparent performance on original data...")
    original_model, original_sess = train_dynamic_deephit_model(
        df_, id_time_status_list,observation, bin_list, cont_list,
        pred_time, eval_time, file_path, hyperparams,
        seed=seed, norm_mode=norm_mode,
        burn_in_mode=burn_in_mode, boost_mode=boost_mode
    )

    # Get apparent performance
    _, (orig_data, orig_time, orig_label), _, orig_data_mi, _ = impt.import_dataset(
        df_, id_time_status_list,observation, bin_list, cont_list, norm_mode
    )
    risk_all = f_get_risk_predictions(original_sess, original_model, orig_data, orig_data_mi, pred_time, eval_time)
    cindex_orig,brier_orig= calculate_metrics(
        risk_all, orig_time, orig_label, orig_time, orig_label,
        pred_time, eval_time,num_Event)

    # 3. Bootstrap validation
    print(f"\nStarting bootstrap validation with {n_bootstrap} iterations...")
    all_c_index = []
    all_brier = []
    for i in range(n_bootstrap):
        try:
            print(f"\nBootstrap iteration {i + 1}/{n_bootstrap}")

            # Generate bootstrap sample
            bootstrap_df = resample(df_, replace=True, random_state=seed + i)

            # Train model on bootstrap sample
            bootstrap_model, bootstrap_sess = train_dynamic_deephit_model(
                bootstrap_df, id_time_status_list, observation, bin_list, cont_list,
                pred_time, eval_time, file_path, hyperparams,
                seed=seed + i, norm_mode=norm_mode,
                burn_in_mode=burn_in_mode, boost_mode=boost_mode
            )

            # Prepare bootstrap and original data
            _, (boot_data, boot_time, boot_label), _, boot_data_mi, _ = impt.import_dataset(
                bootstrap_df, id_time_status_list, observation, bin_list, cont_list, norm_mode
            )

            # Calculate bootstrap performance (on bootstrap sample)
            boot_risk = f_get_risk_predictions(bootstrap_sess, bootstrap_model, boot_data, boot_data_mi, pred_time,
                                               eval_time)
            cindex_boot, brier_boot = calculate_metrics(
                boot_risk, boot_time, boot_label, boot_time, boot_label,
                pred_time, eval_time, num_Event)

            # Calculate test performance (on original data)
            cindex_test, brier_test = calculate_metrics(
                boot_risk, orig_time, orig_label, boot_time, boot_label,
                pred_time, eval_time, num_Event)

            # Calculate optimism for this bootstrap iteration
            cindex_optimism = [a - b for a, b in zip(cindex_boot, cindex_test)]
            brier_optimism = [a - b for a, b in zip(brier_boot, brier_test)]

            all_c_index.append(cindex_optimism)
            all_brier.append(brier_optimism)
        except Exception as e:
            print(f"Error in iteration i={i}: {e}")
            continue

    mean_c_index = [
        np.nanmean([result[event_idx] for result in all_c_index], axis=0)
        for event_idx in range(num_Event)
    ]

    mean_brier = [
        np.nanmean([result[event_idx] for result in all_brier], axis=0)
        for event_idx in range(num_Event)
    ]

    # Calculate optimism-corrected performance
    cindex_corr=[a - b for a, b in zip(cindex_orig, mean_c_index)]
    brier_corr=[a + b for a, b in zip(brier_orig, mean_brier)]
    event_names = [f"Event_{i + 1}" for i in range(num_Event)]

    c_index_mean = {
        event_names[i]: pd.DataFrame(
            cindex_corr[i],
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
            brier_corr[i],
            index=[f"Pred_{t:.1f}" for t in pred_time],
            columns=[f"Eval_{t:.1f}" for t in eval_time]
        ).rename_axis(
            index="Prediction Time",
            columns="Evaluation Time"
        )
        for i in range(num_Event)
    }

    return c_index_mean,brier_mean


# Helper functions



def calculate_metrics(risk_all,va_time, va_label, tr_time, tr_label, pred_time, eval_time,
                      num_Event):
    """Calculate C-index and Brier score for all combinations."""
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
                    va_time,
                    (va_label[:, 0] == event).astype(int),
                    int(t_time) + int(p_time)
                )

                # Calculate Brier score
                brier = weighted_brier_score(
                    tr_time,
                    (tr_label[:, 0] == event).astype(int),
                    risk_all[k][:, p, t],
                    va_time,
                    (va_label[:, 0] == event).astype(int),
                    int(t_time) + int(p_time)
                )
                # Store in matrices
                c_index_values[k][p, t] = c_idx
                brier_values[k][p, t] = brier

    return c_index_values,brier_values


