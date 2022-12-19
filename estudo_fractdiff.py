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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.under_sampling import RandomUnderSampler 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None

def buy_sell_position(bs_list):
    
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

def pred_prob(forest,X,p=0.50):
    X_ = X
    y_pred_prob = forest.predict_proba(X_)
    map_pred = lambda y_prob: np.sum(np.array(y_prob>=p, dtype=int)*[-1,0,1])
    y_pred_t_prob = np.array([map_pred(yi) for yi in y_pred_prob])
    return  y_pred_t_prob

#
start_date = "2006-08-01"
end_date = "2022-12-01"
PETR4 =  yf.download(["PETR4.SA"],start_date,end_date)

list_year = [2016,2018,2018,2019,2020,2021]

year = 2016

for year in [2016,2017,2018,2019,2020,2021]:

    first_date = PETR4.index>=datetime.datetime(year-10,1,1)
    last_date = PETR4.index<datetime.datetime(year+1,1,1)
    #

    df = PETR4[first_date & last_date]
    df.name = 'PETR4'

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
    df['Label'] = labeling.min_max(t_limit, len(df['Close']), 2).shift(0).fillna(0)

    print(df.groupby('Label')['Close'].count())
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

    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_resample(X_train, y_train)
            
    forest = RandomForestClassifier(n_estimators=30,max_depth=9, 
                                    min_samples_leaf=6,
                                    bootstrap=True)
    forest.fit(X_train, y_train)

    print('Train')
    y_pred_t1 = forest.predict(X_train)
    print(accuracy_score(y_train, y_pred_t1))
    print(confusion_matrix(y_train, y_pred_t1))

    print('Test')
    y_pred_t2 =  pred_prob(forest,X_test,p=0.5) #forest.predict(X_test)
    print(accuracy_score(y_test, y_pred_t2))
    print(confusion_matrix(y_test, y_pred_t2))

    print('Teste*')
    y_pred_t_ =  pred_prob(forest,X_,p=0.5) #forest.predict(X_)
    print(accuracy_score(y_, y_pred_t_))
    print(confusion_matrix(y_, y_pred_t_))

    df_['Signal'] = buy_sell_position(np.array(y_pred_t_))
    df_['Label_pred_'] = y_pred_t_ 

    entries = df_['Signal']==1
    exits = df_['Signal']==-1
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('Modelo:')
    print(pf.total_return())
    pf.plot()