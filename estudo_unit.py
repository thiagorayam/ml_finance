import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import Labeling.get_label as lb
import FractDiff.fractdiff as fdiff
import ChartingFunctions.charting as plt_lb
import Utils.pre_process as pp
import vectorbt as vbt
import ta
#https://technical-analysis-library-in-python.readthedocs.io/en/latest/
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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

stock_index = PETR4
year = 2017

print(stock_index.name)
print(year)

first_date = stock_index.index>=datetime.datetime(year-10,1,1)
last_date = stock_index.index<datetime.datetime(year+1,1,1)
#

df = stock_index[first_date & last_date]
df.name = stock_index.name

adf_test = fdiff.MemoryVsCorr(df['Close'], [0,1], 20, 10, 'Close')
d = adf_test[adf_test['adf']<=adf_test['1%'].min()]['order'].min()

n=4

df['Close_'] = fdiff.ts_differencing(df[['Close']],d,10)
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
df.dropna(axis=1, how='all', inplace=True)

t_limit = 7
labeling = lb.Labelling(df['Close'])
df['Label'] = labeling.min_max_v2(t_limit, len(df['Close']), 2).fillna(0)           
df.dropna(inplace=True)
df['Label'] = df['Label'].astype(int)
columns_list = [
                'Close_',
                'momentum_stoch_rsi',
                'momentum_wr',
                'volatility_kcp',
                'volatility_bbp',
                'trend_cci',
                'trend_macd_diff',
                'volume_vpt',
                'volume_fi',
                'others_dlr'
                ]
df = df[columns_list+['Close','Label']]

N = 5
for var in columns_list:
    for x in range(1, N + 1):
        df[var+'_'+str(x)] = df[var].shift(x)

df.dropna(inplace=True)

df_ = df[df.index>=datetime.datetime(year,1,1)]
df_ = df_[t_limit:]
df_t = df[df.index<datetime.datetime(year,1,1)]

X = df_t.loc[:,~df_t.columns.isin(['Close','Label'])]
y = df_t.Label
X_ = df_.loc[:,~df_.columns.isin(['Close','Label'])]
y_ = df_.Label

X_train = X[X.index<datetime.datetime(year - 1,1,1)]
y_train = y[y.index<datetime.datetime(year - 1,1,1)]
X_test = X[X.index>=datetime.datetime(year - 1,1,1)]
y_test = y[y.index>=datetime.datetime(year - 1,1,1)]

scores = []
for random_state in range(1, 101):
    # Train the classifier with the current random state
    clf = RandomForestClassifier(random_state=random_state,
                                class_weight='balanced_subsample',
                                n_estimators=100,
                                max_depth=9,
                                min_samples_leaf=6,
                                bootstrap=True)
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Calculate the accuracy score for the current random state
    score = accuracy_score(y_test, y_pred)
    # Store the accuracy score in the list
    scores.append(score)
best_random_state = np.argmax(scores) + 1


forest = RandomForestClassifier(class_weight='balanced_subsample',
                                n_estimators=100,
                                max_depth=9,
                                min_samples_leaf=6,
                                bootstrap=True,
                                random_state=35)
forest.fit(X_train, y_train)
model = forest

print('Train')
y_pred_t1 = model.predict(X_train)
print(accuracy_score(y_train, y_pred_t1))
print(confusion_matrix(y_train, y_pred_t1))

print('Test')
y_pred_t2 =  model.predict(X_test) 
print(accuracy_score(y_test, y_pred_t2))
print(confusion_matrix(y_test, y_pred_t2))

print('Teste*')
y_pred_t_ =  model.predict(X_) 
print(accuracy_score(y_, y_pred_t_))
print(confusion_matrix(y_, y_pred_t_))


df_['Label_pred_'] = y_pred_t_ 
df_['Label_pred_'] = df_['Label_pred_'].shift(+1).fillna(0)

df_['Signal'] = pp.buy_sell_position(df_.index,np.array(df_['Label_pred_']),df_['Close'],0.1)
entries = df_['Signal']==1
exits = df_['Signal']==-1
pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
print('Modelo:')
print(pf.total_return())
pf.plot()

df_['Signal_'] = pp.buy_sell_position_(np.array(df_['Label_pred_']))
entries = df_['Signal_']==1
exits = df_['Signal_']==-1
pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
print('Modelo2:')
print(pf.total_return())
pf.plot()

plt_lb.plot_label(df_['Close'],df_['Label'])
plt_lb.plot_label(df_['Close'],df_['Label_pred_'])