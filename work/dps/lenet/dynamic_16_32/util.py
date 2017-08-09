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



def noticeEMail(starttime, usr, psw, fromaddr, toaddr):
    """
    Sends an email message through GMail once the script is completed.  
    Developed to be used with AWS so that instances can be terminated 
    once a long job is done. Only works for those with GMail accounts.
    
    starttime : a datetime() object for when to start run time clock
    usr : the GMail username, as a string
    psw : the GMail password, as a string 
    
    fromaddr : the email address the message will be from, as a string
    
    toaddr : a email address, or a list of addresses, to send the 
             message to
    """

    # Calculate run time
    runtime=datetime.now() - starttime
    
    # Initialize SMTP server
    server=smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(usr,psw)
    
    # Send email
    senddate=datetime.strftime(datetime.now(), '%Y-%m-%d')
    subject="Your job has completed"
    m="Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer: My-Mail\r\n\r\n" % (senddate, fromaddr, toaddr, subject)
    msg='''
    
    Job runtime: '''+str(runtime)
    
    server.sendmail(fromaddr, toaddr, m+msg)
    server.quit()

#vis_square((fixed_solver.net.blobs['data']).data[0:10].transpose(0,2,3,1))
# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show(block=False)

def flickrnet(txt, batch_size, train):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(batch_size=batch_size, new_height=256, new_width=256, source=txt,
                             include=dict(phase=train),
                             transform_param=dict(mirror=bool(1-train), crop_size=227,
                             mean_file='data/ilsvrc12/imagenet_mean.binaryproto'), ntop=2)

    n.dataround = L.Round(n.data, i_length=data_i_length, f_length=total_length-data_i_length, in_place=True)

    n.conv1 = L.Convolution(n.dataround, kernel_size=11, num_output=96, stride=4,
                             param=[dict(lr_mult=1, decay_mult=1, i_length=param_conv1_i_length[0], f_length=param_conv1_f_length[0]),
                                    dict(lr_mult=2, decay_mult=0, i_length=param_conv1_i_length[1], f_length=param_conv1_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.01),
                             bias_filler=dict(type='constant', value=0))
    n.conv1round = L.Round(n.conv1, i_length=conv1_i_length, f_length=total_length-conv1_i_length, in_place=True)
    n.relu1 = L.ReLU(n.conv1round, in_place=True)
    n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=0.0001, beta=0.75)

    n.conv2 = L.Convolution(n.norm1, kernel_size=5, num_output=256, pad=2, group=2,
                             param=[dict(lr_mult=1, decay_mult=1, i_length=param_conv2_i_length[0], f_length=param_conv2_f_length[0]),
                                    dict(lr_mult=2, decay_mult=0, i_length=param_conv2_i_length[1], f_length=param_conv2_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.01),
                             bias_filler=dict(type='constant', value=1))
    n.conv2round = L.Round(n.conv2, i_length=conv2_i_length, f_length=total_length-conv2_i_length, in_place=True)
    n.relu2 = L.ReLU(n.conv2round, in_place=True)
    n.pool2 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=0.0001, beta=0.75)

    n.conv3 = L.Convolution(n.norm2, kernel_size=3, num_output=384, pad=1,
                             param=[dict(lr_mult=1, decay_mult=1, i_length=param_conv3_i_length[0], f_length=param_conv3_f_length[0]),
                                    dict(lr_mult=2, decay_mult=0, i_length=param_conv3_i_length[1], f_length=param_conv3_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.01),
                             bias_filler=dict(type='constant', value=0))
    n.conv3round = L.Round(n.conv3, i_length=conv3_i_length, f_length=total_length-conv3_i_length, in_place=True)
    n.relu3 = L.ReLU(n.conv3round, in_place=True)

    n.conv4 = L.Convolution(n.relu3, kernel_size=3, num_output=384, pad=1, group=2,
                             param=[dict(lr_mult=1, decay_mult=1, i_length=param_conv4_i_length[0], f_length=param_conv4_f_length[0]),
                                    dict(lr_mult=2, decay_mult=0, i_length=param_conv4_i_length[1], f_length=param_conv4_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.01),
                             bias_filler=dict(type='constant', value=1))
    n.conv4round = L.Round(n.conv4, i_length=conv4_i_length, f_length=total_length-conv4_i_length, in_place=True)
    n.relu4 = L.ReLU(n.conv4round, in_place=True)

    n.conv5 = L.Convolution(n.relu4, kernel_size=3, num_output=256, pad=1, group=2,
                             param=[dict(lr_mult=1, decay_mult=1, i_length=param_conv5_i_length[0], f_length=param_conv5_f_length[0]),
                                    dict(lr_mult=2, decay_mult=0, i_length=param_conv5_i_length[1], f_length=param_conv5_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.01),
                             bias_filler=dict(type='constant', value=1))
    n.conv5round = L.Round(n.conv5, i_length=conv5_i_length, f_length=total_length-conv5_i_length, in_place=True)
    n.relu5 = L.ReLU(n.conv5round, in_place=True)
    n.pool5 = L.Pooling(n.relu5, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    n.fc6 = L.InnerProduct(n.pool5, num_output=4096,
                             param=[dict(lr_mult=1, decay_mult=1, i_length=param_fc6_i_length[0], f_length=param_fc6_f_length[0]),
                                    dict(lr_mult=2, decay_mult=0, i_length=param_fc6_i_length[1], f_length=param_fc6_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.005),
                             bias_filler=dict(type='constant', value=1))
    n.fc6round = L.Round(n.fc6, i_length=fc6_i_length, f_length=total_length-fc6_i_length, in_place=True)
    n.relu6 = L.ReLU(n.fc6round, in_place=True)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    n.fc7 = L.InnerProduct(n.drop6, num_output=4096,
                             param=[dict(lr_mult=1, decay_mult=1, i_length=param_fc7_i_length[0], f_length=param_fc7_f_length[0]),
                                    dict(lr_mult=2, decay_mult=0, i_length=param_fc7_i_length[1], f_length=param_fc7_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.005),
                             bias_filler=dict(type='constant', value=1))
    n.fc7round = L.Round(n.fc7, i_length=fc7_i_length, f_length=total_length-fc7_i_length, in_place=True)
    n.relu7 = L.ReLU(n.fc7round, in_place=True)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.fc8_flickr = L.InnerProduct(n.drop7, num_output=20,
                             param=[dict(lr_mult=10, decay_mult=1, i_length=param_fc8_i_length[0], f_length=param_fc8_f_length[0]),
                                    dict(lr_mult=20, decay_mult=0, i_length=param_fc8_i_length[1], f_length=param_fc8_f_length[1])],
                             weight_filler=dict(type='gaussian', std=0.01),
                             bias_filler=dict(type='constant', value=0))

    n.fc8_flickrround = L.Round(n.fc8_flickr, i_length=fc8_i_length, f_length=total_length-fc8_i_length, in_place=True)
    n.loss = L.SoftmaxWithLoss(n.fc8_flickrround, n.label)
    n.accuracy = L.Accuracy(n.fc8_flickrround, n.label,
                             include=dict(phase=1))
    return n.to_proto()
    
def flickrnetdata(txt, batch_size, train):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data, n.label = L.ImageData(batch_size=batch_size, new_height=256, new_width=256, source=txt,
                             include=dict(phase=train),
                             transform_param=dict(mirror=bool(1-train), crop_size=227,
                             mean_file='data/ilsvrc12/imagenet_mean.binaryproto'), ntop=2)
    return n.to_proto()

def plot_blob(net, blob, printstats=True, plot=True):
    epsilon = 2**-16 
    stats = dict(absmax=[], std = [], mean=[], ilength = [], margin = [])
    i = 0
    if blob=='blob':
        for blobName, v in net.blobs.items():
            feat = v.data
            stats['absmax'].append( np.max(np.array(abs(feat).flat)) )
            if stats['absmax'][i] < epsilon:
                print 'All features are zeros in layer ' + blobName
                stats['absmax'][i] = epsilon
            stats['std'].append( np.std(np.array(feat.flat)) )
            stats['mean'].append( np.mean(np.array(feat.flat)) )
            stats['ilength'].append( int(np.ceil(np.log2(stats['absmax'][i]))) )
            stats['margin'].append( (2**(stats['ilength'][i])-stats['absmax'][i])/2**(stats['ilength'][i]) )
            #if stats['margin'][i] < 0.25:
            #    stats['ilength'][i] += 1
            #stats['margin'][i] = (2**(stats['ilength'][i])-stats['absmax'][i])/2**(stats['ilength'][i])
            stat_str  = blobName + '_i_length = ' + str(stats['ilength'][i])
            stat_str += ' # ' + blobName + ' ' + str(v.data.shape)
            stat_str += ', max: ' + str(stats['absmax'][i])
            stat_str += ', margin: ' + str(stats['margin'][i])
            stat_str += ', std: ' + str(stats['std'][i])
            stat_str += ', mean: ' + str(stats['mean'][i])
            if printstats == True:
                print stat_str
            if (plot == True):
                scale = feat.size/1000000
                if scale > 0:
                    print 'Too many data in layer ' + blobName + ', it will be shrinked by ' + str(scale)
                    idx = len(feat)/scale
                    feat = feat[0:idx]
                plt.figure(figsize=(10,10))
                plt.subplot(2, 1, 1)
                plt.title(blobName + ' data')
                plt.plot(feat.flat)
                plt.subplot(2, 1, 2)
                try:
                    plt.hist(feat.flat[feat.flat > 0], bins=100)
                except ValueError:  #raised if `y` is empty.
                    print 'Value error is occured in layer ' + blobName
                    pass
                plt.show(block=False)
            i += 1
    else:
        for blobName, v in net.params.items():
            for j in range(2):
                filters = v[j].data
                stats['absmax'].append( np.max(np.array(abs(filters).flat)) )
                if stats['absmax'][i] < epsilon:
                    print 'All parameters are zeros in layer ' + blobName
                    stats['absmax'][i] = epsilon
                stats['std'].append( np.std(np.array(filters.flat)) )
                stats['mean'].append( np.mean(np.array(filters.flat)) )
                stats['ilength'].append( int(np.ceil(np.log2(stats['absmax'][i]))) )
                stats['margin'].append( (2**(stats['ilength'][i])-stats['absmax'][i])/2**(stats['ilength'][i]) )
                stat_str  = 'param_' + blobName + '_i_length[' + str(i) + '] = ' + str(stats['ilength'][i])
                stat_str += ' # ' + blobName + ' ' + str(v[j].data.shape)
                stat_str += ', max: ' + str(stats['absmax'][i])
                stat_str += ', margin: ' + str(stats['margin'][i])
                stat_str += ', std: ' + str(stats['std'][i])
                stat_str += ', mean: ' + str(stats['mean'][i])
                if printstats == True:
                    print stat_str
                if (plot == True):
                    scale = filters.size/1000000
                    if scale > 0:
                        print 'Too many parameters in layer ' + blobName + ', it will be shrinked by ' + str(scale)
                        idx = len(filters)/scale
                        filters = filters[0:idx]
                    plt.figure(figsize=(10,10))
                    plt.subplot(2, 1, 1)
                    plt.title(blobName + ' params[' + str(j) + ']')
                    plt.plot(filters.flat)
                    plt.subplot(2, 1, 2)
                    try:
                        plt.hist(filters.flat[filters.flat > 0], bins=100)
                    except ValueError:  #raised if `y` is empty.
                        print 'Value error is occured in layer ' + blobName
                        pass
                    plt.show(block=False)
                i += 1
    return stats

def get_blob_stats(net, blob, printstats=False, plot=False):
    epsilon = 2**-16 
    stats = dict(absmax=[], std = [], mean=[], ilength = [], tlength = [], margin = [])
    i = 0
    if blob=='blob':
        stat_str = ''
        stat_str2 = ''
        for blobName, v in net.blobs.items():
            feat = v.data
            stats['absmax'].append( np.max(np.array(abs(feat).flat)) )
            if stats['absmax'][i] < epsilon:
                print 'All features are zeros in layer ' + blobName
                stats['absmax'][i] = epsilon
            stats['std'].append( np.std(np.array(feat.flat)) )
            stats['mean'].append( np.mean(np.array(feat.flat)) )
            stats['ilength'].append( int(np.ceil(np.log2(stats['absmax'][i]))) )
            #stats['tlength'].append( int(np.ceil(np.log2(v.data.size))) )
            if v.data.ndim > 0:
                stats['tlength'].append( int(np.ceil(np.log2(v.data.size/v.data.shape[0]))) )
            else:
                stats['tlength'].append(0)
            stats['margin'].append( (2**(stats['ilength'][i])-stats['absmax'][i])/2**(stats['ilength'][i]) )
            #if stats['margin'][i] < 0.25:
            #    stats['ilength'][i] += 1
            #stats['margin'][i] = (2**(stats['ilength'][i])-stats['absmax'][i])/2**(stats['ilength'][i])
            stat_str += blobName + '_i_length = ' + str(stats['ilength'][i])
            stat_str += ' # ' + blobName + ' ' + str(v.data.shape)
            stat_str += ', max: ' + str(stats['absmax'][i])
            stat_str += ', margin: ' + str(stats['margin'][i])
            stat_str += ', std: ' + str(stats['std'][i])
            stat_str += ', mean: ' + str(stats['mean'][i])
            stat_str += '\n'
            stat_str2 += blobName + '_total_length = ' + str(stats['tlength'][i])
            #stat_str2 += ' # initial guess for ' + blobName + ' ' + str(v.data.size)
            stat_str2 += ' # initial guess for ' + blobName + ' ' + str(v.data.size)
            stat_str2 += '\n'
            #if printstats == True:
            #    print stat_str2
            #    print stat_str
            if plot == True:
                scale = feat.size/1000000
                if scale > 0:
                    print 'Too many data in layer ' + blobName + ', it will be shrinked by ' + str(scale)
                    idx = len(feat)/scale
                    feat = feat[0:idx]
                plt.figure(figsize=(10,10))
                plt.subplot(2, 1, 1)
                plt.title(blobName + ' data')
                plt.plot(feat.flat)
                plt.subplot(2, 1, 2)
                try:
                    plt.hist(feat.flat[feat.flat > 0], bins=100)
                except ValueError:  #raised if `y` is empty.
                    print 'Value error is occured in layer ' + blobName
                    pass
                plt.show(block=False)
            i += 1
        if printstats == True:
            print stat_str2
            print stat_str
    else:
        stat_str = ''
        stat_str2 = ''
        for blobName, v in net.params.items():
            for j in range(2):
                filters = v[j].data
                stats['absmax'].append( np.max(np.array(abs(filters).flat)) )
                if stats['absmax'][i] < epsilon:
                    print 'All parameters are zeros in layer ' + blobName
                    stats['absmax'][i] = epsilon
                stats['std'].append( np.std(np.array(filters.flat)) )
                stats['mean'].append( np.mean(np.array(filters.flat)) )
                stats['ilength'].append( int(np.ceil(np.log2(stats['absmax'][i]))) )
                stats['tlength'].append( int(np.ceil(np.log2(v[j].data.size))) )
                stats['margin'].append( (2**(stats['ilength'][i])-stats['absmax'][i])/2**(stats['ilength'][i]) )
                stat_str += 'param_' + blobName + '_i_length[' + str(j) + '] = ' + str(stats['ilength'][i])
                stat_str += ' # ' + blobName + ' ' + str(v[j].data.shape)
                stat_str += ', max: ' + str(stats['absmax'][i])
                stat_str += ', margin: ' + str(stats['margin'][i])
                stat_str += ', std: ' + str(stats['std'][i])
                stat_str += ', mean: ' + str(stats['mean'][i])
                stat_str += '\n'
                stat_str2 += blobName + '_total_length[' + str(j) + '] = ' + str(stats['tlength'][i])
                stat_str2 += ' # initial guess for ' + blobName + ' ' + str(v[j].data.size)
                stat_str2 += '\n'
                #if printstats == True:
                #    print stat_str2
                #    print stat_str
                if plot == True:
                    scale = filters.size/1000000
                    if scale > 0:
                        print 'Too many parameters in layer ' + blobName + ', it will be shrinked by ' + str(scale)
                        idx = len(filters)/scale
                        filters = filters[0:idx]
                    plt.figure(figsize=(10,10))
                    plt.subplot(2, 1, 1)
                    plt.title(blobName + ' params[' + str(j) + ']')
                    plt.plot(filters.flat)
                    plt.subplot(2, 1, 2)
                    try:
                        plt.hist(filters.flat[filters.flat > 0], bins=100)
                    except ValueError:  #raised if `y` is empty.
                        print 'Value error is occured in layer ' + blobName
                        pass
                    plt.show(block=False)
                i += 1
        if printstats == True:
            print stat_str2
            print stat_str
    return stats

def make_dict(net, blob):
    stats = dict()
    if blob=='blob':
        for blobName, v in net.blobs.items():
            stats[blobName + '_absmax'] = []
            stats[blobName + '_std'] = []
            stats[blobName + '_mean'] = []
            stats[blobName + '_ilength'] = []
            stats[blobName + '_margin'] = []
    else:
        for blobName, v in net.params.items():
            for j in range(2):
                stats[blobName + '_absmax_' + str(j)] = []
                stats[blobName + '_std_' + str(j)] = []
                stats[blobName + '_mean_' + str(j)] = []
                stats[blobName + '_ilength_' + str(j)] = []
                stats[blobName + '_margin_' + str(j)] = []
    return stats

def append_dict(net, blob, target_stats, stats):
    i = 0
    if blob=='blob':
        for blobName, v in net.blobs.items():
            target_stats[blobName + '_absmax'].append(stats['absmax'][i])
            target_stats[blobName + '_std'].append(stats['std'][i])
            target_stats[blobName + '_mean'].append(stats['mean'][i])
            target_stats[blobName + '_ilength'].append(stats['ilength'][i])
            target_stats[blobName + '_margin'].append(stats['margin'][i])
            i += 1
    else:
        for blobName, v in net.params.items():
            for j in range(2):
                target_stats[blobName + '_absmax_' + str(j)].append(stats['absmax'][i])
                target_stats[blobName + '_std_' + str(j)].append(stats['std'][i])
                target_stats[blobName + '_mean_' + str(j)].append(stats['mean'][i])
                target_stats[blobName + '_ilength_' + str(j)].append(stats['ilength'][i])
                target_stats[blobName + '_margin_' + str(j)].append(stats['margin'][i])
                i += 1
    return target_stats

def plot_df(df, key_regex="*", title="Blob Statistics", xlabel="Iteration", ylabel="Values"):
    keys = []
    for key in df:
        m = re.search(key_regex, key)
        if m:
            keys.append(key)
    ax = df[keys].plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.show(block=False)

def plot_dict(stats, key_regex="*", title="Blob Statistics", xlabel="Iteration", ylabel="Values"):
    df_stats = pd.DataFrame()
    df_stats = df_stats.from_dict(stats)
    plot_df(df_stats, key_regex, title, xlabel, ylabel)

def plot_three(a, b, c, xlabel="Iteration", ylabel_a="", ylabel_b=""):
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(a.size), a, 'b-')
    ax2.plot(arange(b.size), b, 'r-', arange(c.size), c, 'k-')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_a)
    ax2.set_ylabel(ylabel_b)
    plt.show(block=False)

def plot_twinx(a, b, xlabel="Iteration", ylabel_a="", ylabel_b="", show=False):
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(a.size), a, 'b-')
    ax2.plot(arange(b.size), b, 'r-')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_a)
    ax2.set_ylabel(ylabel_b)
    if show:
        plt.show(block=False)

def plot_lc(train_loss, test_acc, fixed_train_loss=None, fixed_test_acc=None, fixed_train_smoothed_loss=None):
    niter = size(train_loss)
    if size(test_acc) > 1:
        test_interval = (niter-1)/(size(test_acc)-1)
    else:
        teset_interval = 1
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    if fixed_train_loss == None:
        ax1.plot(arange(niter), train_loss, 'b-')
    else:
        if fixed_train_smoothed_loss == None:
            ax1.plot(arange(niter), train_loss, 'b-', arange(niter), fixed_train_loss, 'r-')
        else:
            ax1.plot(arange(niter), train_loss, 'b-', arange(niter), fixed_train_loss, 'r-', arange(niter), fixed_train_smoothed_loss, 'g-')

    if fixed_test_acc == None:
        ax2.plot(test_interval * arange(size(test_acc)), test_acc, 'bs-')
    else:
        ax2.plot(test_interval * arange(size(test_acc)), test_acc, 'bs-', test_interval * arange(size(test_acc)), fixed_test_acc, 'rs-')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    plt.show(block=False)

def get_max_df(df, key_regex="*", printstats=True):
    keys = []
    for key in df:
        m = re.search(key_regex, key)
        if m:
            keys.append(key)
    if printstats == True:
        print df[keys].max()
    return df[keys].max()

def get_final_df(df, key_regex="*", printstats=True):
    keys = []
    for key in df:
        m = re.search(key_regex, key)
        if m:
            keys.append(key)
    if printstats == True:
        print df[keys].iloc[-1]
    return df[keys].iloc[-1]


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

