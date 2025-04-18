_EPSILON = 1e-08
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import dynamic_deephit.utils.import_data as impt
from dynamic_deephit.core.model import Model_Longitudinal_Attention
from dynamic_deephit.core.functions   import c_index,f_get_minibatch, f_get_boosted_trainset


def _f_get_pred(sess, model, data, data_mi, pred_horizon):
    '''
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    '''
    new_data = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))

    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]), np.cumsum(data[:, :, 0], axis=1)[:, :-1]], axis=1)

    for i in range(np.shape(data)[0]):
        last_meas = np.sum(meas_time[i, :] <= pred_horizon)

        new_data[i, :last_meas, :] = data[i, :last_meas, :]
        new_data_mi[i, :last_meas, :] = data_mi[i, :last_meas, :]

    return model.predict(new_data, new_data_mi)


def f_get_risk_predictions(sess, model, data_, data_mi_, pred_time, eval_time):
    pred = _f_get_pred(sess, model, data_[[0]], data_mi_[[0]], 0)
    _, num_Event, num_Category = np.shape(pred)

    risk_all = {}
    for k in range(num_Event):
        risk_all[k] = np.zeros([np.shape(data_)[0], len(pred_time), len(eval_time)])

    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon = int(p_time)
        pred = _f_get_pred(sess, model, data_, data_mi_, pred_horizon)

        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time) + pred_horizon  # if eval_horizon >= num_Category, output the maximum...

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:, :, pred_horizon:(eval_horizon + 1)], axis=2)  # risk score until eval_time
            risk = risk / (np.sum(np.sum(pred[:, :, pred_horizon:], axis=2), axis=1,
                                  keepdims=True) + _EPSILON)  # conditioniong on t > t_pred

            for k in range(num_Event):
                risk_all[k][:, p, t] = risk[:, k]

    return risk_all


def train_dynamic_deephit_model(df_, id_time_status_list, observation,bin_list, cont_list, pred_time, eval_time, file_path, hyperparams, seed=42, norm_mode='standard', burn_in_mode='ON', boost_mode='ON',max_length=None):
    """
    Trains a Dynamic-DeepHit model using the provided DataFrame and hyperparameters.

    Args:
        df_ (pd.DataFrame): The input dataset containing patient information and features.
        id_time_status_list (list): List of column names for patient ID, time, and status.
        observation(list): observation feature column names.
        bin_list (list): List of binary feature column names.
        cont_list (list): List of continuous feature column names.
        pred_time (list): List of prediction time points.
        eval_time (list): List of evaluation time points.
        file_path (str): Path to save the trained model.
        hyperparams (dict): Dictionary of hyperparameters.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        norm_mode (str, optional): Normalization mode for continuous features. Defaults to 'standard'.
        burn_in_mode (str, optional): Whether to perform burn-in training. Defaults to 'ON'.
        boost_mode (str, optional): Whether to boost the training set. Defaults to 'ON'.
        max_length : The maximum number of observations for all IDs in the modeling dataset.

    Returns:
        tuple: A tuple containing the following:
            - model: The trained Dynamic-DeepHit model.
            - sess: The TensorFlow session used by the model.
    """
    # Call import_dataset to process the data
    _, DATA, MASK, data_mi, _ = impt.import_dataset(df_, id_time_status_list,observation, bin_list, cont_list, norm_mode,max_length)

    # Extract processed data
    data, time, label = DATA
    mask1, mask2, mask3 = MASK

    # Calculate num_Event, num_Category, and max_length
    _, num_Event, num_Category = np.shape(mask1)  # dim of mask1: [subj, Num_Event, Num_Category]
    max_length = np.shape(data)[1]  # Maximum length of the time series

    # Define input dimensions
    input_dims = {
        'x_dim': data.shape[2],  # Total feature dimension
        'x_dim_cont': len(cont_list),  # Continuous feature dimension
        'x_dim_bin': len(bin_list),  # Binary feature dimension
        'num_Event': num_Event,  # Number of event types
        'num_Category': num_Category,  # Number of categories
        'max_length': max_length  # Maximum length of the time series
    }

    # Build network_settings
    network_settings = {
        'h_dim_RNN': hyperparams['h_dim_RNN'],  # RNN hidden layer dimension
        'h_dim_FC': hyperparams['h_dim_FC'],  # Fully connected layer hidden dimension
        'num_layers_RNN': hyperparams['num_layers_RNN'],  # Number of RNN layers
        'num_layers_ATT': hyperparams['num_layers_ATT'],  # Number of attention layers
        'num_layers_CS': hyperparams['num_layers_CS'],  # Number of cause-specific layers
        'RNN_type': hyperparams['RNN_type'],  # RNN type (LSTM or GRU)
        'FC_active_fn': hyperparams['FC_active_fn'],  # Activation function for fully connected layers
        'RNN_active_fn': hyperparams['RNN_active_fn'],  # Activation function for RNN layers
        'initial_W': tf.contrib.layers.xavier_initializer(),  # Weight initializer
        'reg_W': hyperparams['reg_W'],  # Regularization coefficient for input layer weights
        'reg_W_out': hyperparams['reg_W_out']  # Regularization coefficient for output layer weights
    }

    # Ensure the directory exists
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Split data into training and validation sets
    (tr_data, va_data, tr_data_mi, va_data_mi, tr_time, va_time, tr_label, va_label,
     tr_mask1, va_mask1, tr_mask2, va_mask2, tr_mask3, va_mask3) = train_test_split(
        data, data_mi, time, label, mask1, mask2, mask3, test_size=0.2, random_state=seed
    )

    # Boost the training set if boost_mode is ON
    if boost_mode == 'ON':
        tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3 = f_get_boosted_trainset(
            tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3
        )

    # Reset the default graph and create a new session
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize the model
    model = Model_Longitudinal_Attention(sess, "Dynamic-DeepHit", input_dims, network_settings)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # Burn-in training
    if burn_in_mode == 'ON':
        print("BURN-IN TRAINING ...")
        for itr in range(hyperparams['iteration_burn_in']):
            x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(
                hyperparams['mb_size'], tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3
            )
            DATA = (x_mb, k_mb, t_mb)
            MISSING = (x_mi_mb)

            _, loss_curr = model.train_burn_in(DATA, MISSING, hyperparams['keep_prob'], hyperparams['lr_train'])

            if (itr + 1) % 1000 == 0:
                print('itr: {:04d} | loss: {:.4f}'.format(itr + 1, loss_curr))

    # Main training
    print("MAIN TRAINING ...")
    min_valid = 0.5

    for itr in range(hyperparams['iteration']):
        x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(
            hyperparams['mb_size'], tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3
        )
        DATA = (x_mb, k_mb, t_mb)
        MASK = (m1_mb, m2_mb, m3_mb)
        MISSING = (x_mi_mb)
        PARAMETERS = (hyperparams['alpha'], hyperparams['beta'], hyperparams['gamma'])

        _, loss_curr = model.train(DATA, MASK, MISSING, PARAMETERS, hyperparams['keep_prob'], hyperparams['lr_train'])

        if (itr + 1) % 1000 == 0:
            print('itr: {:04d} | loss: {:.4f}'.format(itr + 1, loss_curr))

        # Validation (based on average C-index of our interest)
        if (itr + 1) % 1000 == 0:
            risk_all = f_get_risk_predictions(sess, model, va_data, va_data_mi, pred_time, eval_time)

            for p, p_time in enumerate(pred_time):
                pred_horizon = int(p_time)
                val_result1 = np.zeros([input_dims['num_Event'], len(eval_time)])

                for t, t_time in enumerate(eval_time):
                    eval_horizon = int(t_time) + pred_horizon
                    for k in range(input_dims['num_Event']):
                        val_result1[k, t] = c_index(risk_all[k][:, p, t], va_time, (va_label[:, 0] == k + 1).astype(int), eval_horizon)

                if p == 0:
                    val_final1 = val_result1
                else:
                    val_final1 = np.append(val_final1, val_result1, axis=0)

            tmp_valid = np.mean(val_final1)

            if tmp_valid > min_valid:
                min_valid = tmp_valid
                save_path = saver.save(sess, os.path.join(file_path, 'model'))
                print("Model saved in:", save_path)
                print('updated.... average c-index = ' + str('%.4f' % tmp_valid))
    # Return model
    return model,sess