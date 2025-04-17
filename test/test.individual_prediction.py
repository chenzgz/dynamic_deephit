import tensorflow as tf
import pandas as pd
import numpy as np
import unittest
from dynamic_deephit import predict_with_dynamic_deephit
from dynamic_deephit import train_dynamic_deephit_model
from dynamic_deephit import import_dataset,load_test_data
import os
from pathlib import Path

data=load_test_data()
newdata=data[data['id'].isin([5619,5670])]
pred_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
eval_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
id_time_status_list=['id', 'time', 'status']
observation=['yearse']
bin_list=['sex']
con_list=['age','weight','GFR','proteinuria']

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
project_root = Path(__file__).parent.parent
save_dir =(project_root / "test" / "temp").as_posix() + "/"
model,sess=train_dynamic_deephit_model(data, id_time_status_list,observation,bin_list,con_list,pred_time,eval_time,save_dir ,new_parser)
graph = tf.get_default_graph()
input_tensor = graph.get_tensor_by_name("Dynamic-DeepHit/Placeholder_5:0")
max_length = input_tensor.shape[1].value
predictions =predict_with_dynamic_deephit(model,sess,newdata,id_time_status_list,observation,bin_list,con_list,pred_time,eval_time,max_length=max_length)
print(predictions)