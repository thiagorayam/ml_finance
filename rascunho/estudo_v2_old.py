import pandas_datareader as data
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Labeling.get_label as lb
import FractDiff.fractdiff as fdiff
import ChartingFunctions.charting as plt_lb
import ModelValidation.validation_curve as vld
import vectorbt as vbt
import seaborn as sns
import ta
#https://technical-analysis-library-in-python.readthedocs.io/en/latest/
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None

def buy_sell_position_(bs_list):
    
    sit_list = np.zeros(len(bs_list))
    bs_list_final = np.zeros(len(bs_list))

    first_buy = np.where(bs_list == 1)[0][0]
    sit_list[first_buy:] = 1
    bs_list_final[first_buy] = 1

    for idx, x in np.ndenumerate(bs_list):
        index = idx[0]
        index_ = max(idx[0] - 1,0)

        if abs(bs_list[index]-sit_list[index_]) == 2:
            sit_list[index:] = bs_list[index]
            bs_list_final[index] = bs_list[index]
    
    return bs_list_final

def check_buy_group(signal):

    i=0

    buy_ = False
    sell_ = False

    group = np.zeros(len(signal))

    for idx, x in np.ndenumerate(signal):

        index = idx[0]
        signal_atual = signal[index]
        group[index] = i

        buy = signal_atual==1
        sell = signal_atual==-1

        if buy:
            buy_ = True

        if sell:
             sell_ = True

        if buy_ and sell_:
            if buy:
                i = i+1
                group[index] = i
                buy_ = False
                sell_ = False
        
    return group
           
def buy_sell_position(index_df,signal,close,limit=0.05):

    close_ = close.copy()
    signal_ = signal.copy()

    sit_list = np.zeros(len(signal_))
    bs_list_final = np.zeros(len(signal_))
    close_list = np.zeros(len(close_))

    first_buy = np.where(signal_ == 1)[0][0]
    sit_list[first_buy:] = 1
    bs_list_final[first_buy] = 1
    close_list[:] = close_[first_buy]
    last_stop_loss = False

    for idx in np.arange(len(signal_)):

        index = idx
        index_ = max(idx - 1,0)

        lim_cond = (abs(close_[index]-close_list[index_])/close_[index] >= limit)
        bought_cond = sit_list[index_] == 1

        stop_loss = lim_cond and bought_cond

        change_cond = (abs(signal_[index]-sit_list[index_]) == 2) and not last_stop_loss

        if change_cond | stop_loss:
                        
            close_list[index:] = close_[index]

            if stop_loss:
                last_stop_loss = True
                sit_list[index:] = -1
                bs_list_final[index] = -1
            else:
                sit_list[index:] = signal_[index]
                bs_list_final[index] = signal_[index]

        if  signal_[index] == -1  and sit_list[index_] == -1 and last_stop_loss:
            last_stop_loss = False                      

    return pd.Series(data=bs_list_final,index=index_df)

# Setup
start_date = "2006-08-01"
end_date = "2022-12-01"
ITUB4 =  yf.download(["ITUB4.SA"],start_date,end_date)
ITUB4.name = 'ITUB4'
PETR4 =  yf.download(["PETR4.SA"],start_date,end_date)
PETR4.name = 'PETR4'
VALE3 =  yf.download(["VALE3.SA"],start_date,end_date)
VALE3.name = 'VALE3'
BBDC4 =  yf.download(["BBDC4.SA"],start_date,end_date)
BBDC4.name = 'BBDC4'
ABEV3 =  yf.download(["ABEV3.SA"],start_date,end_date)
ABEV3.name = 'ABEV3'

list_stock = [PETR4,VALE3,ITUB4,BBDC4,ABEV3]
list_year = [2017,2018,2019,2020,2021]

df_dict = {}

for stock_index in list_stock:

    df__ = pd.DataFrame()
    for year in list_year:

        print(stock_index.name)
        print(year)

        first_date = stock_index.index>=datetime.datetime(year-10,1,1)
        last_date = stock_index.index<datetime.datetime(year+1,1,1)
        #

        df = stock_index[first_date & last_date]
        df.name = stock_index.name

        adf_test = fdiff.MemoryVsCorr(df['Close'], [0,1], 20, 10, 'Close')
        d=adf_test[adf_test['adf']<=adf_test['1%'].min()]['order'].min()

        n=12

        #
        df['Close_diff'] = fdiff.ts_differencing(df[['Close']],d,10)

        df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        df.dropna(axis=1, how='all', inplace=True)

        N = n
        close_diff_list = []
        for x in range(1, N + 1):
            df['Close_diff_'+str(x)] = df['Close_diff'].shift(x)
            close_diff_list = close_diff_list + ['Close_diff_'+str(x)]
        df.dropna(inplace=True)
        #

        #
        t_limit = 10
        labeling = lb.Labelling(df['Close'])
        df['Label'] = labeling.min_max(t_limit, len(df['Close']), 2).fillna(0)

        #df['Label_1'] = labeling.min_max(t_limit, len(df['Close']), 2).shift(0).fillna(0)
        #df['Label_2'] = labeling.getBinsFromTrend(span=20).fillna(0)

        #plt_lb.plot_label(df['Close'],df['Label_1'])
        #plt_lb.plot_label(df['Close'],df['Label_2'])

        #df['Label'] = 0
        #df.loc[df['Label_1']==df['Label_2'],'Label'] = df.loc[df['Label_1']==df['Label_2'],'Label_1']
        #df.drop(columns=['Label_1','Label_2'],inplace=True)
        
        df.dropna(inplace=True)
        df['Label'] = df['Label'].astype(int)

        #plt_lb.plot_label(df['Close'][-100:],df['Label'][-100:])

        #print(df.groupby('Label')['Close'].count())
        #

        #
        df = df[['Close','momentum_rsi','momentum_wr','volume_mfi','trend_macd']+close_diff_list+['Label']]
        df.dropna(inplace=True)
        df['Label'] = df['Label'].astype(int)
        #

        #
        df_ = df[df.index>=datetime.datetime(year,1,1)]
        df_ = df_[t_limit:]
        df_t = df[df.index<datetime.datetime(year,1,1)]
        #

        X = df_t[df_t.columns[1:-1]]
        y = df_t.Label
        X_ = df_[df_.columns[1:-1]]
        y_ = df_.Label

        X_train = X[X.index<datetime.datetime(year - 1,1,1)]
        y_train = y[y.index<datetime.datetime(year - 1,1,1)]
        X_test = X[X.index>=datetime.datetime(year - 1,1,1)]
        y_test = y[y.index>=datetime.datetime(year - 1,1,1)]

        forest = RandomForestClassifier(class_weight='balanced_subsample',
                                        n_estimators=100,
                                        max_depth=9,
                                        min_samples_leaf=6,
                                        bootstrap=True)
        forest.fit(X_train, y_train)
        model = forest

        print('Train')
        y_pred_t1 = model.predict(X_train)
        print(accuracy_score(y_train, y_pred_t1))
        print(confusion_matrix(y_train, y_pred_t1))

        print('Test')
        y_pred_t2 =  model.predict(X_test) #pred_prob(model,X_test,p=0.65)
        print(accuracy_score(y_test, y_pred_t2))
        print(confusion_matrix(y_test, y_pred_t2))

        print('Teste*')
        y_pred_t_ =  model.predict(X_) #pred_prob(model,X_,p=0.65)
        print(accuracy_score(y_, y_pred_t_))
        print(confusion_matrix(y_, y_pred_t_))

        df_['Label_pred_'] = y_pred_t_ 

        #df_['Signal'] = buy_sell_position_v2(df_.index,np.array(y_pred_t_),df_['Close'])
        #df_['Signal'] = buy_sell_position(np.array(y_pred_t_))
        #plt_lb.plot_label(df_['Close'],df_['Label'])
        #plt_lb.plot_label(df_['Close'],df_['Label_pred_'])
        #plt_lb.plot_label(df_['Close'],df_['Signal'])

        df__ = pd.concat([df__,df_])

    df__['Signal'] = buy_sell_position(df__.index,np.array(df__['Label_pred_']),df__['Close'])
    df_dict[stock_index.name] = df__

result = pd.DataFrame(columns=['ativo','modelo','bandhold'])
i = 0
for stock_name in df_dict.keys():
    print(stock_name)

    stock_name = 'PETR4'

    df__ = df_dict[stock_name]
    result.loc[i,'ativo'] = stock_name


    df_ = df__.drop_duplicates()

    df_['Signal'] = buy_sell_position(df_.index,np.array(df_['Label_pred_']),df_['Close'],0.03)

    entries = df_['Signal']==1
    exits = df_['Signal']==-1
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('Modelo:')
    print(pf.total_return())
    pf.plot()
    result.loc[i,'modelo'] = pf.total_return()


    entries[1:] = False
    entries[0] = True
    exits[:] = False
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('BandH:')
    print(pf.total_return())
    #pf.plot()
    result.loc[i,'bandhold'] = pf.total_return()

    i = i+1

    print(accuracy_score(df_['Label'], df_['Label_pred_']))
    print(confusion_matrix(df_['Label'], df_['Label_pred_']))


df_.columns
result.to_clipboard()