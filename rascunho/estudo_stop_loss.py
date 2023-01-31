import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Labeling.get_label as lb
import FractDiff.fractdiff as fdiff
import ChartingFunctions.charting as plt_lb
import vectorbt as vbt
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import ModelValidation.validation_curve as vld
from Utils import pre_process as pp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None

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

year = 2018
stock_index = PETR4

first_date = stock_index.index>=datetime.datetime(year-10,1,1)
last_date = stock_index.index<datetime.datetime(year+1,1,1)

df = stock_index[first_date & last_date]
df.name = stock_index.name

#########################################################

# X features
adf_test = fdiff.MemoryVsCorr(df['Close'], [0,1], 20, 10, 'Close')
d=adf_test[adf_test['adf']<=adf_test['1%'].min()]['order'].min()
n=12
df['Close_diff'] = fdiff.ts_differencing(df[['Close']],d,10)
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
df.dropna(axis=1, how='all', inplace=True)
N = n
close_diff_list = []
for x in range(1, N + 1):
    df['Close_diff_'+str(x)] = df['Close_diff'].shift(x)
    close_diff_list = close_diff_list + ['Close_diff_'+str(x)]
df.dropna(inplace=True)

# y Label
t_limit = 10
labeling = lb.Labelling(df['Close'])
df['Label_1'] = labeling.min_max(t_limit, len(df['Close']), 2).shift(0).fillna(0)
df['Label_2'] = labeling.getBinsFromTrend(span=20).fillna(0)
df['Label'] = 0
df.loc[df['Label_1']==df['Label_2'],'Label'] = df.loc[df['Label_1']==df['Label_2'],'Label_1']
df.drop(columns=['Label_1','Label_2'],inplace=True)
df.dropna(inplace=True)
df['Label'] = df['Label'].astype(int)

# Split train test
df_ = df[df.index>=datetime.datetime(year,1,1)]
df_ = df_[t_limit:]
df_t = df[df.index<datetime.datetime(year,1,1)]

X = df_t[df_t.columns[1:-1]]
y = df_t.Label
X_ = df_[df_.columns[1:-1]]
y_ = df_.Label

X_train = X[X.index<datetime.datetime(year - 1,1,1)]
y_train = y[y.index<datetime.datetime(year - 1,1,1)]
X_test = X[X.index>=datetime.datetime(year - 1,1,1)]
y_test = y[y.index>=datetime.datetime(year - 1,1,1)]

# Training
def optimize():
    test_range = np.arange(1,15,1)
    model = RandomForestClassifier(class_weight='balanced_subsample'
                                            ,n_estimators=30
                                            ,min_samples_leaf=6
                                            ,max_depth=10
                                                )
    vld.plot_validation_curve(X,y,model,test_range,
                                "max_depth","accuracy")


forest = RandomForestClassifier(class_weight='balanced_subsample',
                                n_estimators=100,
                                max_depth=9,
                                min_samples_leaf=6,
                                bootstrap=True)
forest.fit(X_train, y_train)
model = forest

# Result
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

# Financial result
### TREINO
df_t = df_t[df_t.index<datetime.datetime(year - 1,1,1)]
df_t['Label_pred_'] = y_pred_t1
df_t['Signal'] = pp.buy_sell_position(np.array(y_pred_t1))
plt_lb.plot_label(df_t['Close'],df_t['Label_pred_'])
plt_lb.plot_label(df_t['Close'],df_t['Signal'])
entries = df_t['Signal']==1
exits = df_t['Signal']==-1
pf = vbt.Portfolio.from_signals(df_t['Close'], entries, exits)
print('Modelo:')
print(pf.total_return())
pf.plot()

df_t['Signal'] = pp.buy_sell_position(np.array(df_t['Label']))
plt_lb.plot_label(df_t['Close'],df_t['Label'])
plt_lb.plot_label(df_t['Close'],df_t['Signal'])
entries = df_t['Signal']==1
exits = df_t['Signal']==-1
pf = vbt.Portfolio.from_signals(df_t['Close'], entries, exits)
print('Modelo ótimo:')
print(pf.total_return())
pf.plot()
#################

df_['Label_pred_'] = y_pred_t_
df_['Signal'] = pp.buy_sell_position(np.array(y_pred_t_))
plt_lb.plot_label(df_['Close'],df_['Label_pred_'])
plt_lb.plot_label(df_['Close'],df_['Signal'])
entries = df_['Signal']==1
exits = df_['Signal']==-1
pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
print('Modelo:')
print(pf.total_return())
pf.plot()

df_['Signal'] = pp.buy_sell_position(np.array(df_['Label']))
plt_lb.plot_label(df_['Close'],df_['Label'])
plt_lb.plot_label(df_['Close'],df_['Signal'])
entries = df_['Signal']==1
exits = df_['Signal']==-1
pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
print('Modelo ótimo:')
print(pf.total_return())
pf.plot()