import os
#os.chdir("..")
import sys
sys.path.insert(0, './python')
import caffe
import re

from pylab import *
from caffe import layers as L
from caffe import params as P
import matplotlib.pyplot as plt
import pandas as pd
from util import *

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

def fixed_lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)
    n.dataround = L.Round(n.data, i_length=data_i_length, f_length=data_f_length, in_place=True)
    n.conv1 = L.Convolution(n.dataround, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'),
                             param=[dict(i_length=param_conv1_i_length[0], f_length=param_conv1_f_length[0]),
                                    dict(i_length=param_conv1_i_length[1], f_length=param_conv1_f_length[1])])
    n.conv1round = L.Round(n.conv1, i_length=conv1_i_length, f_length=conv1_f_length, in_place=True)
    n.pool1 = L.Pooling(n.conv1round, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'),
                             param=[dict(i_length=param_conv2_i_length[0], f_length=param_conv2_f_length[0]),
                                    dict(i_length=param_conv2_i_length[1], f_length=param_conv2_f_length[1])])
    n.conv2round = L.Round(n.conv2, i_length=conv2_i_length, f_length=conv2_f_length, in_place=True)
    n.pool2 = L.Pooling(n.conv2round, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'),
                             param=[dict(i_length=param_ip1_i_length[0], f_length=param_ip1_f_length[0]),
                                    dict(i_length=param_ip1_i_length[1], f_length=param_ip1_f_length[1])])
    n.ip1round = L.Round(n.ip1, i_length=ip1_i_length, f_length=ip1_f_length, in_place=True)
    n.relu1 = L.ReLU(n.ip1round, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'),
                             param=[dict(i_length=param_ip2_i_length[0], f_length=param_ip2_f_length[0]),
                                    dict(i_length=param_ip2_i_length[1], f_length=param_ip2_f_length[1])])
    #n.ip2round = L.Round(n.ip2, i_length=ip2_i_length, f_length=ip2_f_length, in_place=True)
    #n.loss = L.SoftmaxWithLoss(n.ip2round, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

if __name__ == "__main__":

    plt.close('all')
        
    caffe_root = os.environ.get('CAFFE_ROOT') + '/'
    caffe.set_mode_gpu()
    niter = 1001
    float_niter = float(niter)
    test_iters = 100
    test_interval =50 
    # losses will also be stored in the log
    train_loss = zeros(niter)
    all_train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(float_niter / test_interval)))
    all_test_acc = zeros(int(np.ceil(float_niter / test_interval)))
    fixed_train_loss = zeros(niter)
    fixed_train_smoothed_loss = zeros(niter)
    window = 100
    fixed_test_acc = zeros(int(np.ceil(float_niter / test_interval)))
    
    ############################################################################################################
    train_loss = np.genfromtxt('ref_train_loss.csv', delimiter=",")
    test_acc = np.genfromtxt('ref_test_acc.csv', delimiter=",")
    all_train_loss = train_loss
    all_test_acc = test_acc
    ############################################################################################################
    
    total_length = 31 # 32 - 1 (sign bit)
    # Total bit width initialization for the parameters
    param_conv1_total_length = [total_length, total_length]
    param_conv2_total_length = [total_length, total_length]
    param_ip1_total_length   = [total_length, total_length]
    param_ip2_total_length   = [total_length, total_length]

    # Initial guess for the integer part bit width for the parameters
    param_conv1_i_length = [0, 0]
    param_conv2_i_length = [0, 0]
    param_ip1_i_length   = [0, 0]
    param_ip2_i_length   = [0, 0]

    # Initial guess for the fractional part bit width for the parameters
    param_conv1_f_length = [total_length, total_length]
    param_conv2_f_length = [total_length, total_length]
    param_ip1_f_length   = [total_length, total_length]
    param_ip2_f_length   = [total_length, total_length]
    
    # Total bit width initialization for the neurons
    data_total_length = total_length
    conv1_total_length = total_length
    conv2_total_length = total_length
    ip1_total_length = total_length

    # Initial guess for the integer part bit width for the neurons
    data_i_length = 8
    label_i_length = 8
    conv1_i_length = 8
    pool1_i_length = 8
    conv2_i_length = 8
    pool2_i_length = 8
    ip1_i_length = 8
    ip2_i_length = 8
    loss_i_length = -3

    # Initial guess for the fractional part bit width for the neurons
    data_f_length  = data_total_length  - data_i_length 
    conv1_f_length = conv1_total_length - conv1_i_length
    conv2_f_length = conv2_total_length - conv2_i_length
    ip1_f_length   = ip1_total_length   - ip1_i_length  
    
    ####################################
    # write prototxt for fixed point
    ####################################
    with open('fixed_lenet_auto_train.prototxt', 'w') as f:
        f.write(str(fixed_lenet(caffe_root + 'examples/mnist/mnist_train_lmdb', 64)))
        
    with open('fixed_lenet_auto_test.prototxt', 'w') as f:
        f.write(str(fixed_lenet(caffe_root + 'examples/mnist/mnist_test_lmdb', 100)))
    
    fixed_solver = caffe.SGDSolver('./fixed_lenet_auto_solver.prototxt')
    fixed_solver_blob_stats = make_dict(fixed_solver.net, 'blob')
    fixed_solver_param_stats = make_dict(fixed_solver.net, 'param')

    ####################################
    # the main solver loop
    ####################################
    for it in range(niter):
        fixed_solver.step(1)  # SGD by Caffe
        
        # store the train loss
        fixed_train_loss[it] = fixed_solver.net.blobs['loss'].data
        if it == 0:
            fixed_train_smoothed_loss[it] = fixed_train_loss[it]
        elif it < window:
            fixed_train_smoothed_loss[it] = (fixed_train_smoothed_loss[it-1]*it+fixed_train_loss[it])/(it+1)
        else:
            fixed_train_smoothed_loss[it] = fixed_train_smoothed_loss[it-1] + (fixed_train_loss[it] -fixed_train_loss[it-window])/window
        
        # run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        #  how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(test_iters):
                fixed_solver.test_nets[0].forward()
                correct += sum(fixed_solver.test_nets[0].blobs['ip2'].data.argmax(1)
                               == fixed_solver.test_nets[0].blobs['label'].data)
            fixed_test_acc[it // test_interval] = correct / 1e4

            stats = get_blob_stats(fixed_solver.net, 'blob')
            fixed_solver_blob_stats = append_dict(fixed_solver.net, 'blob', fixed_solver_blob_stats, stats)
            stats = get_blob_stats(fixed_solver.net, 'param')
            fixed_solver_param_stats = append_dict(fixed_solver.net, 'param', fixed_solver_param_stats, stats)
    
    ####################################
    # Plot Learning Curve
    ####################################
    fig_name = 'lenet_loss_param_t' + '.png'
#    plot_lc(fixed_train_smoothed_loss, test_acc, fixed_train_loss, fixed_test_acc)

    loss_csv = np.genfromtxt('fixed_lenet_loss.csv', delimiter=",")
    total_length = loss_csv[:,6]
    plot_twinx(fixed_train_smoothed_loss, total_length, ylabel_a="fixed_train_smoothed_loss", ylabel_b="total_length")
    plt.savefig(fig_name)
    
    ####################################
    # Save loss and accuracy
    ####################################
    all_train_loss = np.column_stack((all_train_loss, fixed_train_loss))
    all_test_acc = np.column_stack((all_test_acc, fixed_test_acc))
    np.savetxt("train_loss.csv", all_train_loss, delimiter=",", fmt='%s')
    np.savetxt("fixed_train_smoothed_loss.csv", fixed_train_smoothed_loss, delimiter=",", fmt='%s')
    np.savetxt("test_acc.csv", all_test_acc, delimiter=",", fmt='%s')

    df_stats = pd.DataFrame()
    df_fixed_solver_blob_stats = df_stats.from_dict(fixed_solver_blob_stats)
    df_fixed_solver_blob_stats.to_csv('df_fixed_solver_blob_stats_t' + '.csv')
    df_fixed_solver_param_stats = df_stats.from_dict(fixed_solver_param_stats)
    df_fixed_solver_param_stats.to_csv('df_fixed_solver_param_stats_t' + '.csv')

    ####################################
    # Final blob print and plot
    ####################################

    print '# fixed solver blobs, total_length: '
    stats = get_blob_stats(fixed_solver.net, 'blob', True, False)
    print '# fixed solver params, total_length: '
    stats = get_blob_stats(fixed_solver.net, 'param', True, False)
    
