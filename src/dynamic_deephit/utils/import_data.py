import pandas as pd
import numpy as np


##### USER-DEFINED FUNCTIONS

def f_get_Normalization(X, norm_mode):
    """
    Normalizes the input data matrix based on the specified normalization mode.

    Args:
        X (np.ndarray): Input data matrix with shape (num_Patient, num_Feature).
        norm_mode (str): Normalization mode to apply. Supported modes are 'standard' and 'normal'.

    Returns:
        np.ndarray: Normalized data matrix with the same shape as the input.

    Raises:
        None: Prints an error message if an unsupported normalization mode is provided.
    """
    num_Patient, num_Feature = np.shape(X)  # Get the dimensions of the input data

    if norm_mode == 'standard':  # Zero mean and unit variance normalization
        for j in range(num_Feature):
            if np.nanstd(X[:, j]) != 0:  # Check if the standard deviation is non-zero
                X[:, j] = (X[:, j] - np.nanmean(X[:, j])) / np.nanstd(X[:, j])  # Standardize the feature
            else:
                X[:, j] = (X[:, j] - np.nanmean(X[:, j]))  # Center the feature if std is zero
    elif norm_mode == 'normal':  # Min-max normalization (scaling to [0, 1])
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.nanmin(X[:, j])) / (np.nanmax(X[:, j]) - np.nanmin(X[:, j]))  # Scale the feature
    else:
        print("INPUT MODE ERROR!")  # Print an error message for unsupported modes

    return X  # Return the normalized data matrix


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    """
    Generates a mask for calculating the conditional probability denominator in survival analysis.

    Args:
        meas_time (np.ndarray): Array of last measurement times for each patient, with shape (N, 1).
        num_Event (int): Number of event types (excluding censoring).
        num_Category (int): Number of time intervals or categories.

    Returns:
        np.ndarray: A 3D mask array with shape (N, num_Event, num_Category), where:
                    - N is the number of patients.
                    - 1's are filled up to the last measurement time for each patient.
                    - 0's are filled after the last measurement time.

    Notes:
        This mask is used to calculate the denominator part of the conditional probability
        in survival analysis models.
    """
    # Initialize a 3D mask array with zeros
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category])  # Shape: (N, num_Event, num_Category)

    # Fill the mask with 1's up to the last measurement time for each patient
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0] + 1)] = 1  # Set 1's until the last measurement time

    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    """
    Generates a mask for calculating the log-likelihood loss in survival analysis.

    Args:
        time (np.ndarray): Array of event or censoring times for each patient, with shape (N, 1).
                           Each value represents the time of the event or the last observed time for censored patients.
        label (np.ndarray): Array of event labels for each patient, with shape (N, 1).
                           - 0 indicates censoring (event time is unknown, only known to be after the censoring time).
                           - Non-zero values indicate the event type (e.g., 1, 2, etc.).
        num_Event (int): Number of event types (excluding censoring).
        num_Category (int): Number of time intervals or categories.

    Returns:
        np.ndarray: A 3D mask array with shape (N, num_Event, num_Category), where:
                    - N is the number of patients.
                    - For uncensored patients: one element is set to 1 (indicating the event time and type).
                    - For censored patients: elements are set to 1 after the censoring time (for all events).

    Notes:
        This mask is used to compute the log-likelihood loss in survival analysis models.
        - For uncensored patients, the mask identifies the exact event time and type.
        - For censored patients, the mask accounts for the fact that the event time is only known to be after the censoring time.
    """
    # Initialize a 3D mask array with zeros
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category])  # Shape: (N, num_Event, num_Category)

    # Fill the mask based on event or censoring status
    for i in range(np.shape(time)[0]):
        if label[i, 0] != 0:  # Not censored (event occurred)
            # Set 1 at the specific event time and type
            mask[i, int(label[i, 0] - 1), int(time[i, 0])] = 1
        else:  # Censored (event time unknown, only known to be after the censoring time)
            # Fill 1 for all events after the censoring time
            mask[i, :, int(time[i, 0] + 1):] = 1

    return mask

def f_get_fc_mask3(time, meas_time, num_Category):
    """
    Generates a mask for calculating the ranking loss in survival analysis, used for pair-wise comparisons.

    Args:
        time (np.ndarray): Array of event or censoring times for each patient, with shape (N, 1).
                           Each value represents the time of the event or the last observed time for censored patients.
        meas_time (np.ndarray): Array of last measurement times for each patient, with shape (N, 1).
                                - For longitudinal measurements: Represents the time of the last observation.
                                - For single measurements: Can be empty or ignored.
        num_Category (int): Number of time intervals or categories.

    Returns:
        np.ndarray: A 2D mask array with shape (N, num_Category), where:
                    - N is the number of patients.
                    - For longitudinal measurements:
                        - 1's are set from the last measurement time (exclusive) to the event time (inclusive).
                    - For single measurements:
                        - 1's are set from the start to the event time (inclusive).

    Notes:
        This mask is used to compute the ranking loss in survival analysis models.
        - For longitudinal measurements, the mask focuses on the time interval between the last measurement and the event time.
        - For single measurements, the mask covers the entire period up to the event time.
    """
    # Initialize a 2D mask array with zeros
    mask = np.zeros([np.shape(time)[0], num_Category])  # Shape: (N, num_Category)

    if np.shape(meas_time):  # Longitudinal measurements
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0])  # Last measurement time
            t2 = int(time[i, 0])  # Event or censoring time
            # Set 1's from the last measurement time (exclusive) to the event time (inclusive)
            mask[i, (t1 + 1):(t2 + 1)] = 1
    else:  # Single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0])  # Event or censoring time
            # Set 1's from the start to the event time (inclusive)
            mask[i, :(t + 1)] = 1

    return mask

##### TRANSFORMING DATA
def f_construct_dataset(df, feat_list, id_col, time_col, yearse_col, status_col,max_length=None):
    """
        Parameters:
            df         : Input DataFrame containing the dataset.
            feat_list  : List of feature column names to be used in the analysis.
            id_col     : Column name for patient ID.
            time_col   : Column name for event time (e.g., time-to-event).
            yearse_col : Column name for observation time (e.g., age at the last measurement).
            status_col : Column name for event status (e.g., event indicator).
            max_length : The maximum number of observations for all IDs in the modeling dataset.

        Returns:
            pat_info   : Matrix containing patient information (ID, event time, status, observation time, etc.).
            data       : Matrix containing the processed data for further analysis.
    """
    grouped = df.groupby([id_col])
    id_list = pd.unique(df[id_col])
    if max_length is None:
        max_meas = np.max(grouped.count())[0]
    else:
        max_meas = max_length


    data = np.zeros([len(id_list), max_meas, len(feat_list) + 1])
    pat_info = np.zeros([len(id_list), 5])

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i, 4] = tmp.shape[0]  # number of measurement
        pat_info[i, 3] = np.max(tmp[yearse_col])  # last measurement time
        pat_info[i, 2] = tmp[status_col][0]  # cause
        pat_info[i, 1] = tmp[time_col][0]  # time_to_event
        pat_info[i, 0] = tmp[id_col][0]

        data[i, :int(pat_info[i, 4]), 1:] = tmp[feat_list]
        data[i, :int(pat_info[i, 4] - 1), 0] = np.diff(tmp[yearse_col])

    return pat_info, data
def import_dataset(df_, id_time_status_list,observation ,bin_list, cont_list, norm_mode='standard',max_length=None):
    """
    Processes and prepares a dataset for further analysis or modeling.

    Args:
        df_ (pd.DataFrame): The input dataset containing patient information and features.
        id_time_status_list (list): List of column names for patient ID, time, and status.
        observation(list): observation feature column names.
        bin_list (list): List of binary feature column names.
        cont_list (list): List of continuous feature column names.
        norm_mode (str, optional): Normalization mode for continuous features. Defaults to 'standard'.
        max_length : The maximum number of observations for all IDs in the modeling dataset.

    Returns:
        tuple: A tuple containing the following elements:
            - DIM (tuple): Dimensions of the processed data (x_dim, x_dim_cont, x_dim_bin).
            - DATA (tuple): Processed data including normalized features, time, and labels.
            - MASK (tuple): Masks for handling missing values and event types.
            - data_mi (np.ndarray): Binary mask indicating missing values in the data.
            - pat_info (np.ndarray): Patient information including ID, time, status, and last measurement age.
    """
    # Initialize empty lists if bin_list or cont_list are not provided
    if not bin_list:
        bin_list = []
    if not cont_list:
        cont_list = []

    # Combine binary and continuous feature lists
    feat_list = cont_list + bin_list

    # Extract relevant columns from the dataset
    df_ = df_[id_time_status_list +observation +feat_list]
    df_org_ = df_.copy(deep=True)  # Create a deep copy of the original data

    # Normalize continuous features
    df_.loc[:, cont_list] = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    # Construct datasets from processed and original data
    pat_info, data = f_construct_dataset(df_, feat_list,id_col=id_time_status_list[0], time_col=id_time_status_list[1], yearse_col=observation[0], status_col=id_time_status_list[2],max_length=max_length)
    _, data_org = f_construct_dataset(df_org_, feat_list, id_col=id_time_status_list[0], time_col=id_time_status_list[1], yearse_col=observation[0], status_col=id_time_status_list[2],max_length=max_length)

    # Handle missing values
    data_mi = np.zeros(np.shape(data))  # Binary mask for missing values
    data_mi[np.isnan(data)] = 1  # Mark missing values as 1
    data_org[np.isnan(data)] = 0  # Replace missing values with 0 in the original data
    data[np.isnan(data)] = 0  # Replace missing values with 0 in the processed data

    # Calculate dimensions of the data
    x_dim = np.shape(data)[2]  # Total number of features (1 + x_dim_cont + x_dim_bin)
    x_dim_cont = len(cont_list)  # Number of continuous features
    x_dim_bin = len(bin_list)  # Number of binary features

    # Extract patient information
    last_meas = pat_info[:, [3]]  # Age at the last measurement
    label = pat_info[:, [2]]  # Event labels (competing risks)
    time = pat_info[:, [1]]  # Age when the event occurred

    # Define the number of categories and events
    num_Category = int(np.max(pat_info[:, 1]) * 1.2)  # Define a larger number of categories than the max time-to-event
    num_Event = len(np.unique(label)) - 1  # Number of unique events (excluding 0)

    # Convert to single-risk if only one event type exists
    if num_Event == 1:
        label[np.where(label != 0)] = 1  # Convert all non-zero labels to 1

    # Generate masks for handling missing values and events
    mask1 = f_get_fc_mask1(last_meas, num_Event, num_Category)  # Mask for last measurement
    mask2 = f_get_fc_mask2(time, label, num_Event, num_Category)  # Mask for event time and label
    mask3 = f_get_fc_mask3(time, -1, num_Category)  # Mask for time-to-event

    # Organize output data
    DIM = (x_dim, x_dim_cont, x_dim_bin)  # Dimensions of the data
    DATA = (data, time, label)  # Processed data
    MASK = (mask1, mask2, mask3)  # Masks for handling missing values and events

    return DIM, DATA, MASK, data_mi, pat_info