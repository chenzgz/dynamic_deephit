import tensorflow as tf
import pandas as pd
import numpy as np
import unittest
from dynamic_deephit import external_validation
from dynamic_deephit import train_dynamic_deephit_model
from dynamic_deephit import import_dataset

data=load_test_data()
pred_time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # prediction time (in months)
eval_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
new_parser = {'mb_size': 32,
             'iteration_burn_in': 20,
             'iteration': 100,

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
feat_list=['sex']+['age','weight','GFR','proteinuria']

project_root = Path(__file__).parent.parent
save_dir =(project_root / "test" / "temp").as_posix() + "/"
#import_dataset(data,id_time_status_list,observation,['sex'],['age','weight','GFR','proteinuria'])
model,sess=train_dynamic_deephit_model(data, ['id', 'time', 'status'],['yearse'],['sex'],['age','weight','GFR','proteinuria'],pred_time,eval_time,save_dir,new_parser)
graph = tf.get_default_graph()
input_tensor = graph.get_tensor_by_name("Dynamic-DeepHit/Placeholder_5:0")  # 替换为实际输入名称
max_length = input_tensor.shape[1].value

results=external_validation(model,sess,data, data,id_time_status_list,observation, bin_list, con_list, pred_time, eval_time,max_length=max_length)
print(results)