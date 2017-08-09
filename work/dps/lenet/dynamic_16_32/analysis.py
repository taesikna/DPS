#import os
#os.chdir("..")
#import sys
#sys.path.insert(0, './python')
import re
import caffe
from caffe import layers as L
from caffe import params as P

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smtplib
from datetime import datetime
from util import *

if __name__ == "__main__":

    
    ####################################
    # User defined parameters
    ####################################
    csv_dir = './'
    
    #plt.close('all')

    ####################################
    # Read loss and accuracy
    ####################################
    all_train_loss = np.genfromtxt(csv_dir + 'train_loss.csv', delimiter=",")
    all_test_acc = np.genfromtxt(csv_dir + 'test_acc.csv', delimiter=",")
    loss_csv = np.genfromtxt(csv_dir + 'loss.csv', delimiter=",")
    train_loss = all_train_loss[:,0]
    fixed_train_loss = all_train_loss[:,1]
    test_acc = all_test_acc[:,0]
    fixed_test_acc = all_test_acc[:,1]

    dynamic_mode = loss_csv[:,2]
    fixed_train_smoothed_loss = loss_csv[:,3]
    total_length = loss_csv[:,4]

    niter = train_loss.shape[0]
    print 'niter = ', niter
    test_interval = (niter-1)/(test_acc.shape[0]-1)
    print 'test_interval = ', test_interval

    ####################################
    # Read blob and param dataframe
    ####################################
    df_solver_blob_stats = pd.read_csv(csv_dir + 'df_solver_blob_stats.csv')
    df_solver_param_stats = pd.read_csv(csv_dir + 'df_solver_param_stats.csv')
    df_fixed_solver_blob_stats = pd.read_csv(csv_dir + 'df_fixed_solver_blob_stats_t.csv')
    df_fixed_solver_param_stats = pd.read_csv(csv_dir + 'df_fixed_solver_param_stats_t.csv')

    ####################################
    # plot learning curve
    ####################################
    plot_lc(train_loss, test_acc, fixed_train_loss, fixed_test_acc)
    plot_lc(train_loss, test_acc, fixed_train_loss, fixed_test_acc, fixed_train_smoothed_loss)
    plot_twinx(total_length, dynamic_mode, ylabel_a="total_length", ylabel_b="dynamic_mode")
    plot_twinx(fixed_train_loss, dynamic_mode, ylabel_a="fixed_train_loss", ylabel_b="dynamic_mode")
    plot_twinx(fixed_train_loss, total_length, ylabel_a="fixed_train_loss", ylabel_b="total_length")
    plot_twinx(fixed_train_smoothed_loss, total_length, ylabel_a="fixed_train_smoothed_loss", ylabel_b="total_length")
    plot_three(fixed_train_smoothed_loss, total_length, dynamic_mode, ylabel_a="fixed_train_smoothed_loss", ylabel_b="total_length")

