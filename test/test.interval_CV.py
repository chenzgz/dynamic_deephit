import tensorflow as tf
import pandas as pd
import numpy as np
import unittest
from dynamic_deephit.core.individual_prediction import predict_with_dynamic_deephit
from dynamic_deephit.core.main  import train_dynamic_deephit_model
from dynamic_deephit import import_dataset
from dynamic_deephit.core.main import f_get_risk_predictions,_f_get_pred
from dynamic_deephit.core.interval_CV import interval_CV,k_fold_cross_validation,train_test_split_validation,bootstrap_validation
data=load_test_data()
pred_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # prediction time (in months)
eval_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
new_parser = {'mb_size': 32,

             'iteration_burn_in': 200,
             'iteration': 1000,

             'keep_prob': 0.6,
             'lr_train': 1e-4,

             'h_dim_RNN': 100,
             'h_dim_FC' : 100,
             'num_layers_RNN':2,
             'num_layers_ATT':2,
             'num_layers_CS' :2,

             'RNN_type':'GRU',

             'FC_active_fn' : tf.nn.relu,
             'RNN_active_fn': tf.nn.tanh,

            'reg_W'         : 1e-5,
            'reg_W_out'     : 0.,

             'alpha' :1.0,
             'beta'  :0.1,
             'gamma' :1.0
}
id_time_status_list=['id', 'time', 'status']
observation=['yearse']
bin_list=['sex']
con_list=['age','weight','GFR','proteinuria']
project_root = Path(__file__).parent.parent
save_dir =(project_root / "test" / "temp").as_posix() + "/"
result=interval_CV("K-fold",data,id_time_status_list,observation,bin_list,con_list,pred_time,eval_time,save_dir,new_parser,K=5)
print(result)
#result=k_fold_cross_validation(data,id_time_status_list,observation,bin_list,con_list,pred_time,eval_time,"E:/Program/动态预测/python_packages/test/",new_parser,K=5,n=3)
#print(result)
#result=train_test_split_validation(data,id_time_status_list,observation,bin_list,con_list,pred_time,eval_time,"E:/Program/动态预测/python_packages/test/",new_parser,n=2)
#print(result)
#result=bootstrap_validation(data,id_time_status_list,observation,bin_list,con_list,pred_time,eval_time,"E:/Program/动态预测/python_packages/test/",new_parser,n_bootstrap=5)
#print(result)