"""
This code is provided for the Reservoir Computing Model for Tipping Prediction 
with the help of Shirin Panahi, Ling-wei Kong, and Zhengmeng Zhai.

Usage:
The code has two phases: 
  1. 'Hyperparameter_optimization'  
  2. 'Training'

In the Hyperparameter_optimization phase, the code performs Bayesian Optimization 
to find the optimized hyperparameters for the Reservoir Computing model. 
The optimized hyperparameters are saved in the 'hyperparameter' folder.

In the Training phase:
  - Load the time-series data from the 'data' folder.
  - Initialize the Reservoir Computing (RC) model using the configuration parameters.
  - Train the RC model and validate its predictions.

Make sure to adjust configuration settings before running the model.

"""

import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
from aux_funcs import *
from bayes_opt import BayesianOptimization
import pickle

name = 'CESM_data' # CESM_data, amoc_2d
path = 'data/'+ name + '.pkl'
res_path = f"data"
res_path_hyper = f"hyperparameter"
phase = 'Training'                      #'Hyperparameter_optimization'  # 'Training'


if not os.path.exists(res_path):
    os.makedirs(res_path)

if not os.path.exists(path):
    print('Generating data. This might take a bit!')
    try: 
        data_gen(name)
        print('Data generated succesfuly!')
        df = pd.read_pickle(path)
        print('Data Loaded succesfuly!')
    except Exception as e:
        print("Model doesn't exist!" )
else:
    print('Loading previously generated data')
    df = pd.read_pickle(path)
    print('Data Loaded succesfuly!')
    


if phase == 'Hyperparameter_optimization':
    
    if not os.path.exists(res_path_hyper):
        os.makedirs(res_path_hyper)
    
    def target_amoc(d, rho, gamma, alpha, beta, bias, iter_time=10, proportion=0.8):
        
        # config['n'] = 300
        # config['d'] = opt_params['d']
        # config['alpha'] = opt_params['alpha']
        # config['beta'] = opt_params['beta']
        # config['gamma'] = opt_params['gamma']
        # config['rho'] = opt_params['rho']
        # config['bias'] = opt_params['bias']

        # config['train_length'] = 1650
        # config['wash_length'] = 100
        # config['vali_length'] = 450
        # config['input_dim'] = 2
        # config['output_dim'] = 1

        # config = {}
        # config['n'] = 500
        # config['d'] = d
        # config['alpha'] = alpha
        # config['beta'] = beta
        # config['gamma'] = gamma
        # config['rho'] = rho
        # config['bias'] = bias
        
        # config['train_length'] = 1200 - np.random.randint(100)
        # config['wash_length'] = 0
        # config['vali_length'] = 5
        # config['input_dim'] = 2
        # config['output_dim'] = 1
        
        rmse_all = []
        for i in range(iter_time):
            rc_opt = rc.Reservoir(data=data, config=config, obs_dim=1, Win_type=1, forecast=True)
            rc_opt.data_preprocessing()
            rc_opt.initialize_rc()
            train_preditions, train_x = rc_opt.train()
            rmse, vali_real, vali_pred = rc_opt.validation(vali_step=config['vali_length'])
            
            rmse_all.append(np.mean(rmse))
        
        rmse_mean = np.average(sorted(rmse_all)[:int(proportion * iter_time)])
        
        print(rmse_mean)

        return 1 / rmse_mean


    # # amoc
    data = np.transpose(np.array(df))
    n_iter=200
    
    optimizer = BayesianOptimization(target_amoc,
                                      {'d': (0.01, 1), 'rho': (0.01, 3), 'gamma': (0.01, 5), 'alpha': (0.01, 1), 'beta': (-8, -1), 'bias': (-5, 5)},)

    optimizer.maximize(n_iter)
    print('amoc')
    print(optimizer.max)

    pkl_file = open('./hyperparameter/rc_opt_' + name + '.pkl', 'wb')
    pickle.dump(optimizer.max, pkl_file)
    pkl_file.close()


if phase == 'Training':
    data = np.transpose(np.array(df))
    
    
    data_amoc, param= data[:, 0], data[:, 1]
    
    fig, ax = plt.subplots(2, 1, figsize=(8,6))
    ax0, ax1 = ax[0], ax[1]
    
    ax0.plot(data_amoc)
    ax1.plot(param)
    
    ax0.set_ylabel('data_amoc')
    ax1.set_ylabel('param')
    ax1.set_xlabel('t')
    plt.show()
    
    pkl_file = open('./hyperparameter/rc_opt_' + name + '.pkl', 'rb')
    opt_results = pickle.load(pkl_file)
    pkl_file.close()
    opt_params = opt_results['params']

    
    config = {}
    
    
    # #### amoc_2d
    # config['n'] = 300
    # config['d'] = opt_params['d']
    # config['alpha'] = opt_params['alpha']
    # config['beta'] = opt_params['beta']
    # config['gamma'] = opt_params['gamma']
    # config['rho'] = opt_params['rho']
    # config['bias'] = opt_params['bias']
    
    # config['train_length'] = 1350
    # config['wash_length'] = 5
    # config['vali_length'] = 500
    # config['input_dim'] = 2
    # config['output_dim'] = 1
    
    #### CESM_data 
    config['n'] = 300
    config['d'] = opt_params['d']
    config['alpha'] = opt_params['alpha']
    config['beta'] = opt_params['beta']
    config['gamma'] = opt_params['gamma']
    config['rho'] = opt_params['rho']
    config['bias'] = opt_params['bias']

    config['train_length'] = 1650
    config['wash_length'] = 100
    config['vali_length'] = 450
    config['input_dim'] = 2
    config['output_dim'] = 1
    
    rc = rc.Reservoir(data=data, config=config, obs_dim=1, Win_type=1, forecast=True)
    rc.data_preprocessing()
    rc.initialize_rc()
    train_preditions, train_x = rc.train()
    rmse, vali_real, vali_pred = rc.validation(vali_step=config['vali_length'])
    
    fig, ax = plt.subplots(2, 1, figsize=(8,6))
    ax0, ax1 = ax[0], ax[1]
    
    ax0.plot(train_preditions[0, :], label='data_amoc real')
    ax0.plot(train_x[0, config['wash_length']:], label='data_amoc train', linestyle='--')
    ax0.legend()
    
    ax1.plot(vali_real[:, 0], label='data_amoc real')
    ax1.plot(vali_pred[:, 0], label='data_amoc pred', linestyle='--')
    ax1.legend()
    plt.legend()
    plt.show()



    
