import numpy as np
import random
import scipy.stats as stats
import scipy.sparse as sparse
import networkx as nx
import scipy
import os.path
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.ndimage import gaussian_filter1d
import pandas as pd



class Reservoir:
    def __init__(self, data=None, config=None, Win_type=1, obs_dim=None, forecast=True):
        self.data = data
        self.config = config
        self.Win_type = Win_type
        self.obs_dim = obs_dim
        self.forecast = forecast
        
        # reservoir setting
        self.n = config['n']
        self.d = config['d']
        self.rho = config['rho']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.beta = 10 ** config['beta']
        self.bias = config['bias']
        self.train_length = config['train_length']
        self.wash_length = config['wash_length']
        self.vali_length = config['vali_length']
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        
        self.vali_strat = self.train_length + 1

    def data_preprocessing(self):
        # random_start = np.random.randint(10000, 50000)
        random_start = 0
        self.data = self.data[random_start:, :]
        
        # # normalization
        # scaler = MinMaxScaler()
        # self.data = scaler.fit_transform(self.data)

    def initialize_rc(self):
        if self.Win_type == 1:
            self.Win = np.random.uniform(-self.gamma, 
                                         self.gamma, (self.n, self.input_dim))
        elif self.Win_type == 2:
            self.Win = np.zeros((self.n, self.input_dim))
            n_win = self.n - self.n % self.input_dim
            index = list(range(n_win))
            random.shuffle(index)
            index = np.reshape(index, (int(n_win/self.input_dim), self.input_dim))
            for di in range(self.input_dim):
                self.Win[index[:, di], di] = np.random.uniform(-self.gamma, 
                                             self.gamma, (int(n_win/self.input_dim), 1)).reshape(1, -1)
            aaa = 1
        
        graph = nx.erdos_renyi_graph(self.n, self.d, 42, False)
        for (u, v) in graph.edges():
            graph.edges[u, v]['weight'] = np.random.normal(0.0, 1.0)
        self.A = nx.adjacency_matrix(graph).todense()
        rho = max(np.linalg.eig(self.A)[0])
        self.A = (self.rho / abs(rho)) * self.A

    def train(self):
        r_train = np.zeros((self.n, self.train_length - self.wash_length))
        y_train = np.zeros((self.output_dim, self.train_length - self.wash_length))
        self.r_end = np.zeros((self.n, 1))
        
        train_x = - np.ones((self.train_length, self.input_dim))
        train_y = np.zeros((self.train_length, self.output_dim))
        
        if self.forecast:
            train_y[:, :] = self.data[1:self.train_length+1, :self.output_dim]
        else:
            train_y[:, :] = self.data[:self.train_length, :self.output_dim]
        
        # for om_idx in range(self.train_length):
        #     obs_index = list(range(self.input_dim))
        #     random.shuffle(obs_index)
        #     obs_index = obs_index[:self.obs_dim]
            
        train_x[:, :] = self.data[:self.train_length, :]

        train_x = np.transpose(train_x)
        train_y = np.transpose(train_y)
        
        r_all = np.zeros((self.n, self.train_length+1))
        for ti in range(self.train_length):
            r_all[:, ti+1] = (1 - self.alpha) * r_all[:, ti] + \
                self.alpha * np.tanh( np.dot(self.A, r_all[:, ti]) + np.dot(self.Win, train_x[:, ti]) + self.bias * np.ones((1, self.n))  )
        
        r_out = r_all[:, self.wash_length+1:]
        self.r_end[:] = r_all[:, -1].reshape(-1, 1)
        
        r_train[:, :] = r_out
        y_train[:, :] = train_y[:self.output_dim, self.wash_length:]
        
        self.Wout = np.dot(np.dot(y_train, np.transpose(r_train)), np.linalg.inv(np.dot(r_train, np.transpose(r_train)) + self.beta * np.eye(self.n)) )
        
        train_preditions = np.dot(self.Wout, r_out)
        return train_preditions, train_x
    
    def validation(self, vali_step=1, obs=None, obs_type='self', rend=True):
        # obs_type: will the machine get the input of the obs or not.
        vali_pred = np.zeros((self.vali_length, self.output_dim))
        vali_real = np.zeros((self.vali_length, self.output_dim))
        # if self.vali_strat + self.vali_length <= np.shape(self.data)[0]:
        #     vali_real[:, :] = self.data[self.vali_strat:self.vali_strat + self.vali_length, :]
        # else:
        #     vali_real[:np.shape(self.data[self.vali_strat:, :])[0], :] = self.data[self.vali_strat:, :]
        
        vali_real[:, :] = self.data[self.vali_strat:self.vali_strat + self.vali_length, :self.output_dim]
        parameter_channel = self.data[self.vali_strat:self.vali_strat + self.vali_length, -1]
        if not rend:
            self.r_end = np.zeros((self.n, 1))
        r = self.r_end
        u = -np.ones((self.input_dim, 1))
        
        u[:] = self.data[self.vali_strat-1, :].reshape(-1, 1)
        
        for ti in range(self.vali_length):
            r = (1 - self.alpha) * r + self.alpha * np.tanh(np.dot(self.A, r) + np.dot(self.Win, u) + self.bias * np.ones((self.n, 1)))
            r_out = r
            pred = np.dot(self.Wout, r_out)
            
            vali_pred[ti, :] = pred.reshape(1, -1)
            
            u = -np.ones((self.input_dim, 1))
            if ti % vali_step == 0:
                u[:] = self.data[self.vali_strat+ti, :].reshape(-1, 1)
                aaa = 1
            else:
                u[:self.output_dim] = pred[:].reshape(-1, 1)
                u[-1] = self.data[self.vali_strat+ti, -1]
        
        rmse = rmse_calculation(vali_pred, vali_real)
        self.vali_strat = self.vali_strat + self.vali_length
        self.r_end[:] = r
        
        return rmse, vali_real, vali_pred

def rmse_calculation(A, B):
    # calculate rmse
    return (np.sqrt(np.square(np.subtract(A, B)).mean()))

def mse_calculation(A, B):
    # actual we calculate mse
    return (np.square(np.subtract(A, B)).mean())



