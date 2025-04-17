import dynamic_deephit
import pandas as pd
import tensorflow as tf
import numpy as np
from dynamic_deephit import import_dataset
import unittest
from dynamic_deephit import train_dynamic_deephit_model
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
print(data['yearse'])
id_time_status_list=['id', 'time', 'status']
observation=['yearse']
bin_list=['sex']
con_list=['age','weight','GFR','proteinuria']

project_root = Path(__file__).parent.parent
save_dir =(project_root / "test" / "temp").as_posix() + "/"
#import_dataset(data,id_time_status_list,observation,['sex'],['age','weight','GFR','proteinuria'])
model,sess=train_dynamic_deephit_model(data, ['id', 'time', 'status'],['yearse'],['sex'],['age','weight','GFR','proteinuria'],pred_time,eval_time,save_dir ,new_parser)
