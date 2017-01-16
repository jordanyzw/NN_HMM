#!/usr/bin/env python
# coding=utf-8
from hmmlearn.hmm import GaussianHMM
import MLP
import numpy as np
from matplotlib import cm,pyplot as plt
import localfile_data_util
from matplotlib.dates import DateFormatter
import pandas as pd
import talib
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_variable_hist(X,label):
    plt.hist(X, 20, normed=1, facecolor='green',histtype='barstacked')
    plt.savefig(label + '.png')
    return

def data_preprocess(ticker,n_com,train_pro):
    filename = "../../If_index/" + ticker + "00_2010-01-01_2016-11-20.csv"

    quotes = localfile_data_util.read_local_file(filename)
    dates_list = pd.to_datetime(quotes['date'])
    open_list = np.array(quotes['open'])
    close_list = np.array(quotes['close'])
    high_list = np.array(quotes['high'])
    low_list = np.array(quotes['low'])
    volume_list = np.array(quotes['volume'])

    #features
    logDel = np.log(high_list + [1E-3]*len(high_list)) - np.log(low_list)
    close_diff = np.diff(close_list)
    open_diff = np.diff(open_list)
    high_diff = np.diff(high_list)
    low_diff = np.diff(low_list)
    logRet_5 = np.log(close_list[5:]) - np.log(close_list[:-5])
    logVol_5 = np.log(volume_list[5:]) - np.log(volume_list[:-5])
    Avg_close = talib.SMA(close_list,6)
    close_avg_diff = np.log(close_list[5:]) - np.log(Avg_close[5:])
    alpha_101 = ((close_list - open_list)/((high_list - low_list) + [1E-3] * len(low_list)))
    #make the features have the same length 
    logDel = logDel[5:]
    close_diff = close_diff[4:]
    open_diff = open_diff[4:]
    high_diff = high_diff[4:]
    low_diff = low_diff[4:]
    close_list = close_list[5:]
    Date = dates_list[5:]
    alpha_101 = alpha_101[5:]

    #A = np.column_stack([logDel,logVol_5,logRet_5,close_avg_diff,alpha_101])
    A = np.column_stack([logRet_5,close_avg_diff,alpha_101])
    nnFeatures =np.column_stack([logDel, close_diff, open_diff, high_diff,\
                                 low_diff, alpha_101, logRet_5, logVol_5, close_avg_diff])
    label_list,state_dict,start_index = hmm_state_predict(\
        A,Date,close_diff,close_list,"without_rescaled_", ticker, train_pro,nnFeatures)

def hmm_state_predict(A, Date, close_diff, close_list, figLabel, ticker, train_pro, nnFeatures):
    daysFmt = DateFormatter("%m-%d-%Y")


    #seperate the data
    start_index = int(train_pro * len(A))
    train,test = A[0 : start_index],A[start_index : ]
    train_Date,test_Date = Date[0 : start_index],Date[start_index:]
    train_close_list, test_close_list = close_list[0 : start_index],close_list[start_index:]
    train_close_diff,test_close_diff = close_diff[0 : start_index],close_diff[start_index:]


    model = GaussianHMM(n_components = n_com, covariance_type = "full", n_iter = 5000).fit(train)
    hidden_states = model.predict(train)
    label_list = np.zeros((len(hidden_states), model.n_components))

    for index in range(len(train)):
        label_list[index][hidden_states[index]] = 1
    def price(x):
        return '$1.2f'%x
    fig = plt.figure(figsize=(30,20))
    fig1 = fig.add_subplot(311)
    for i in range(model.n_components):
        pos = (hidden_states == i)
        fig1.plot_date(train_Date[pos], train_close_list[pos], 'o', label='hidden_state %d'%i, lw=2)
        fig1.legend(loc="upper left")
    fig1.plot_date(train_Date,train_close_list,'-')
    fig1.xaxis.set_major_formatter(daysFmt)
    fig1.fmt_xdata = DateFormatter('%Y-%m-%d') 
    fig1.fmt_ydata = price  
    fig1.autoscale_view()
    
    fig2 = fig.add_subplot(312)
    res = pd.DataFrame({'Date' : train_Date,'close_diff':train_close_diff, "state" : hidden_states})
    state_dict = {}
    for i in range(model.n_components):
        pos = (hidden_states == i)
        pos = np.append(0, pos[:-1])#转化为0,1列表
        df = res.close_diff
        res['state_ret%s'%i] = df.multiply(pos)
        fig2.plot_date(train_Date, res['state_ret%s'%i].cumsum(), '-', label='hidden_state %d'%i)
        fig2.legend(loc="upper left")
        #decide what action to make based on the cumulative reward of the train data
        if(list(res['state_ret%s'%i].cumsum())[-1] > 1):
            state_dict['%s'%i] = 1
        else:
            state_dict['%s'%i] = -1

     
    fig2.xaxis.set_major_formatter(daysFmt)
    fig2.fmt_xdata = DateFormatter('%Y-%m-%d') 
    fig2.fmt_ydata = price 
    fig2.autoscale_view()
    plt.savefig(figLabel + ticker + "_" + str(n_com) +  "components_.png")
    del fig,fig1,fig2,res
    #use neural network to predict the action(buy,sell)
    predict_label = MLP.Neural(label_list,state_dict,start_index,nnFeatures,9)
    stateCnt = {}
    for state in predict_label:
        stateCnt[state] = stateCnt.get(state,0) + 1
    print stateCnt
    test_state_list = np.array(predict_label)
    test_profit = []
    result_array = np.zeros((model.n_components, len(test))) 
    print result_array.shape
    for index,ele in enumerate(test):
        
        st = test_state_list[index]
        state = str(st)
        
        if state_dict[state] == 1:
            #print 'bull',ele,state,test_profit[-1]
            test_profit.append(test_close_diff[index])
            result_array[int(st)][index] = 1
        else:
            test_profit.append( -1 * test_close_diff[index])
            #print'bear', ele,state,test_profit[-1]
            result_array[int(st)][index] = -1


    fig = plt.figure(figsize=(25,18))
    fig1 = fig.add_subplot(211)
    
    for i in range(model.n_components):
        pos = (test_state_list == i)
        fig1.plot_date(test_Date[pos], test_close_list[pos], 'o', label='hidden_state %d'%i, lw=2)
        fig1.legend(loc="upper right")
    fig1.plot_date(test_Date,test_close_list,'-')
    fig1.xaxis.set_major_formatter(daysFmt)
    fig1.fmt_xdata = DateFormatter('%Y-%m-%d') 
    fig1.fmt_ydata = price  
    fig1.autoscale_view()
    
    res = pd.DataFrame({'Date':test_Date,"close_diff":test_close_diff})
    fig2 = fig.add_subplot(212)
    for i in range(model.n_components):
        df = res.close_diff
        if (state_dict[str(i)]) == 1:
            print 'bull %s'%i
            res['state_ret%s'%i] = df.multiply(result_array[i])
        else:
            print 'bear %s'%i
            res['state_ret%s'%i] = df.multiply( -1 * result_array[i])

        fig2.plot_date(test_Date, res['state_ret%s'%i].cumsum(), '-', label = 'hidden_state %d'%i, lw=2)
        fig2.legend(loc="upper left")

    fig2.xaxis.set_major_formatter(daysFmt)
    fig2.autoscale_view()
    fig2.fmt_xdata = DateFormatter('%Y-%m-%d') 
    fig2.fmt_ydata = price
    fig2.grid(True)
    plt.savefig(figLabel + ticker + "_" + str(n_com) +  "components_test.png")
    return label_list,state_dict,start_index
    

if __name__ == "__main__":
    import sys
    import os
    ticker = sys.argv[1]
    n_com = int(sys.argv[2])
    train_pro = float(sys.argv[3])
    if not os.path.exists(ticker + "_hmm_state_predict"):
        os.mkdir(ticker + "_hmm_state_predict")
    os.chdir(ticker + "_hmm_state_predict")
    data_preprocess(ticker,n_com,train_pro)






