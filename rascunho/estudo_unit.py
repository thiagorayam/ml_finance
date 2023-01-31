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
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

        up_lim_cond = ((close_[index]-close_list[index_])/close_[index] >= 2*limit)
        low_lim_cond = ((close_[index]-close_list[index_])/close_[index] <= -limit)

        lim_cond = up_lim_cond or low_lim_cond

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
end_date = "2023-01-01"

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
list_year = [2017,2018,2019,2020,2021,2022]

stock_index = BBDC4
year = 2020

print(stock_index.name)
print(year)

first_date = stock_index.index>=datetime.datetime(year-10,1,1)
last_date = stock_index.index<datetime.datetime(year+1,1,1)
#

df = stock_index[first_date & last_date]
df.name = stock_index.name

adf_test = fdiff.MemoryVsCorr(df['Close'], [0,1], 20, 10, 'Close')
d=adf_test[adf_test['adf']<=adf_test['1%'].min()]['order'].min()

n=4

#
#df['Close_diff'] = fdiff.ts_differencing(df[['Close']],d,10)

df['Close_'] = fdiff.ts_differencing(df[['Close']],d,10)
#df['Open_'] = fdiff.ts_differencing(df[['Open']],d,10)
#df['High_'] = fdiff.ts_differencing(df[['High']],d,10)
#df['Low_'] = fdiff.ts_differencing(df[['Low']],d,10)

#df['Return']  = df['Close']

df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
df.dropna(axis=1, how='all', inplace=True)

'''
df['close_'] = fdiff.ts_differencing(df[['Close']],d,10)
N = n
close_diff_list = []
for x in range(1, N + 1):
    df['close_diff_'+str(x)] = df['close_'].shift(x)
    close_diff_list = close_diff_list + ['close_diff_'+str(x)]
df.dropna(inplace=True)
'''
'''
df['high_'] = fdiff.ts_differencing(df[['High']],d,10)
N = n
high_diff_list = []
for x in range(1, N + 1):
    df['high_diff_'+str(x)] = df['high_'].shift(x)
    high_diff_list = high_diff_list + ['high_diff_'+str(x)]
df.dropna(inplace=True)

df['low_'] = fdiff.ts_differencing(df[['Low']],d,10)
N = n
low_diff_list = []
for x in range(1, N + 1):
    df['low_diff_'+str(x)] = df['low_'].shift(x)
    low_diff_list = low_diff_list + ['low_diff_'+str(x)]
df.dropna(inplace=True)

df['open_'] = fdiff.ts_differencing(df[['Open']],d,10)
N = n
open_diff_list = []
for x in range(1, N + 1):
    df['open_diff_'+str(x)] = df['open_'].shift(x)
    open_diff_list = open_diff_list + ['open_diff_'+str(x)]
df.dropna(inplace=True)
'''

#close_list = []
#for x in range(1, N + 1):
#    df['Close_'+str(x)] = df['Close'].shift(x)
#    close_list = close_list + ['Close_'+str(x)]
#df.dropna(inplace=True)

#return_list = []
#for x in range(1, N + 1):
#    df['Return_'+str(x)] = df['Return'].shift(x)
#    return_list = return_list + ['Return_'+str(x)]
#df.dropna(inplace=True)
#

#
#df['Return'] = df['Close'].pct_change().fillna(0)

t_limit = 8
labeling = lb.Labelling(df['Close'])
df['Label'] = labeling.min_max_v2(t_limit, len(df['Close']), 2).fillna(0)
#df.drop(columns='Return',inplace=True)

#df['Label_1'] = labeling.min_max(t_limit, len(df['Close_']), 2).shift(0).fillna(0)
#df['Label_2'] = labeling.getBinsFromTrend(span=20).fillna(0)

#plt_lb.plot_label(df['Close'],df['Label_1'])
#plt_lb.plot_label(df['Close'],df['Label_2'])

#df['Label'] = 0
#df.loc[df['Label_1']==df['Label_2'],'Label'] = df.loc[df['Label_1']==df['Label_2'],'Label_1']
#df.drop(columns=['Label_1','Label_2'],inplace=True)

df.dropna(inplace=True)
df['Label'] = df['Label'].astype(int)
#df['Label'] = buy_sell_position_(df['Label'])

#plt_lb.plot_label(df['Close'][-100:],df['Label'][-100:])

#print(df.groupby('Label')['Close'].count())
#

#

#list_feat = ['trend_cci', 'volatility_bbp',
#            'momentum_wr', 'momentum_stoch_rsi',
#            'volume_vpt', 'others_dr'] 
#df = df[['Close']+list_feat+close_diff_list+['Label']]
df.dropna(inplace=True)

corr_list = df.corr()['Label']
corr_list = corr_list.abs()
corr_list.sort_values(inplace=True)
columns_list = list(corr_list[-20:].index)

if not columns_list.count('Close_'):
    columns_list = columns_list + ['Close_']

columns_list.remove('Label')    

#df = df[['Close']+columns_list+close_diff_list]#+open_diff_list+high_diff_list+low_diff_list]

N = n
for var in columns_list:
    for x in range(1, N + 1):
        df[var+'_'+str(x)] = df[var].shift(x)

df.dropna(inplace=True)
#

#
df_ = df[df.index>=datetime.datetime(year,1,1)]
df_ = df_[t_limit:]
df_t = df[df.index<datetime.datetime(year,1,1)]
#

X = df_t.loc[:,~df_t.columns.isin(['Label'])]
y = df_t.Label

X_ = df_.loc[:,~df_.columns.isin(['Label'])]

#X_['Close_test'] = fdiff.ts_differencing(X_[['Close']],d,10)
X_['Close_test'] = X_['Close'].pct_change().fillna(0)
labeling_test = lb.Labelling(X_['Close_test'])
X_['Label'] = labeling_test.min_max(t_limit, len(X_['Close_test']), 2).fillna(0)
y_ = X_['Label']
X_.drop(columns=['Close_test','Label'],inplace=True)

X_train = X[X.index<datetime.datetime(year - 1,1,1)]
y_train = y[y.index<datetime.datetime(year - 1,1,1)]
X_test = X[X.index>=datetime.datetime(year - 1,1,1)]
y_test = y[y.index>=datetime.datetime(year - 1,1,1)]


#pca = PCA(n_components=20)

# Fit PCA on the training data
#pca.fit(X_train)

# Transform the training and test data
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)
#X_ = pca.transform(X_)

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
df_['Signal'] = buy_sell_position(df_.index,np.array(df_['Label_pred_']),df_['Close'],0.05)

entries = df_['Signal']==1
exits = df_['Signal']==-1
pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
print('Modelo:')
print(pf.total_return())
pf.plot()

df_['Signal'] = buy_sell_position(df_.index,np.array(df_['Label']),df_['Close'],0.1)
entries = df_['Signal']==1
exits = df_['Signal']==-1
pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
print('Modelo:')
print(pf.total_return())
pf.plot()

plt_lb.plot_label(df_['Close'],df_['Label'])
plt_lb.plot_label(df_['Close'],df_['Label_pred_'])