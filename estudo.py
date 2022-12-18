import pandas_datareader as data
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Labeling.get_label as lb
import FractDiff.fractdiff as fdiff
import ChartingFunctions.charting as plt_lb
import vectorbt as vbt
import seaborn as sns
from ta import add_all_ta_features #https://technical-analysis-library-in-python.readthedocs.io/en/latest/
from backtesting import Backtest, Strategy
from backtesting.test import GOOG

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None

def train_forest_model(X,y,n):
    X_train_, X_test, y_train_, y_test = train_test_split(X, y, stratify=y, random_state=0)
    rus = RandomUnderSampler(random_state=0)
    X_train, y_train = rus.fit_resample(X_train_, y_train_)
    #X_train, y_train = X_train_, y_train_
    feature_names = X.columns
    forest = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=3, bootstrap=True)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances = forest_importances.sort_values()
    list_features = forest_importances.sort_values()[-n:].index
    return forest,X_train,y_train,X_test,y_test,list_features,forest_importances

def pred_prob(forest,X,p=0.50):
    X_ = X
    y_pred_prob = forest.predict_proba(X_)
    map_pred = lambda y_prob: np.sum(np.array(y_prob>=p, dtype=int)*[-1,0,1])
    y_pred_t_prob = np.array([map_pred(yi) for yi in y_pred_prob])
    return  y_pred_t_prob

data_source = 'yahoo'
start_date = '2006-08-01'
end_date = '2022-11-29'

PETR4 = data.DataReader('PETR4.SA', data_source, start_date, end_date)
PETR4.name = 'PETR4'
VALE3 = data.DataReader('VALE3.SA', data_source, start_date, end_date)
VALE3.name = 'VALE3'
ITUB4 = data.DataReader('ITUB4.SA', data_source, start_date, end_date)
ITUB4.name = 'ITUB4'
BBDC4 = data.DataReader('BBDC4.SA', data_source, start_date, end_date)
BBDC4.name = 'BBDC4'
ABEV3 = data.DataReader('ABEV3.SA', data_source, start_date, end_date)
ABEV3.name = 'ABEV3'

list_df = [PETR4,VALE3,ITUB4,BBDC4,ABEV3]

df_teste = PETR4.copy()
df_teste.name = 'PETR4'

def plot_label_v2(close,label,name,size_):

    plt.rcParams['xtick.labelsize'] = size_
    plt.rcParams['ytick.labelsize'] = size_   
    plt.figure(figsize = (20,10))
    plt.plot(close,color='k',label=name)

    # long position
    long_cond = close.index.isin(label[(label==1)].index)
    plt.plot(close[long_cond].index, close[long_cond], '^',
                markersize = size_, color='g', label='Compra')

    # short position
    short_cond = close.index.isin(label[(label==-1)].index)
    plt.plot(close[short_cond].index, close[short_cond], 'v',
                markersize = size_, color='r', label='Venda')
                
    plt.xlabel('Data', fontsize = size_ )
    plt.legend( fontsize = size_ )
    plt.grid()
    plt.show()

def test(df_teste):

    df = df_teste.copy()
    print(df_teste.name)

    df['Return'] = df['Close'].diff()

    fdiff.MemoryVsCorr(df['Close'], [0,1], 10, 10, 'Close')

    df['Close_diff'] = fdiff.ts_differencing(df[['Close']],0.2,10)

    data_ = df[['Close','Close_diff','Return']]
    data_.columns = ['d=0.0','d=0.2','d=1.0']
    fig,ax = plt.subplots(1,figsize=(12,4))
    sns.lineplot(data=data_[-500:])
    ax.set_title('Série de fechamento PETR4')
    ax.set(xlabel='Data', ylabel='R$')
    plt.legend(title='Diferenciação', loc='best')
    plt.show()

    df['Return'] = df['Close'].diff()
    df['Close_diff'] = fdiff.ts_differencing(df[['Close']],0.2,10)
    df['High_diff'] = fdiff.ts_differencing(df[['High']],0.2,10)
    df['Low_diff'] = fdiff.ts_differencing(df[['Low']],0.2,10)
    df['Open_diff'] = fdiff.ts_differencing(df[['Open']],0.2,10)
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    df['Return_diff'] = df['Close_diff'].diff()
    df['pct_Return_diff'] = df['Close_diff'].pct_change().fillna(0)
    df['Close_diff_mean_10'] =  df['Close_diff'].rolling(10).mean()
    df['Close_diff_mean_20'] =  df['Close_diff'].rolling(20).mean()
    df['Mov_Avg_Sign_diff'] = df['Close_diff'].rolling(10).mean() - df['Close_diff'].rolling(20).mean()
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)

    N = 8
    for x in range(1, N + 1):
        df['Return_'+str(x)] = df['Return_diff'].shift(x)
    for x in range(1, N + 1):
        df['Close_diff_'+str(x)] = df['Close_diff'].shift(x)
    df['day_of_week'] = df.index.day_of_week
    #

    t_limit = 8
    labeling = lb.Labelling(df['Close_diff'])
    df['Label_1'] = labeling.min_max(t_limit, len(df['Close_diff']), 2).shift(0).fillna(0)
    daily_vol = labeling.getDailyVol()
    df['Label_2'] = labeling.tbm(0.05, 0.05, 8, f=1)
    df.dropna(inplace=True)
    filter_label = df['Label_1'] == df['Label_2']
    df['Label'] = 0
    df.loc[filter_label,'Label'] = df.loc[filter_label,'Label_1']
    df.drop(columns=['Label_1','Label_2'],inplace=True)

    df.groupby('Label')['Close_diff'].count()
    data_ = df.copy()
    data_.index.name = 'Data'
    data_['Fechameneto'] = data_['Close']
    plot_label_v2(df['Close'][0:40],df['Label'][0:40],'Preço PETR4 (R$)',15)

    #df = df[['High','Low','Open','Close','Close_diff','momentum_stoch_rsi','Close_diff_1', 'Close_diff_2','Label']]
    #df = df[['High','Low','Open','Close','Close_diff','momentum_stoch_rsi','volatility_kcp','Close_diff_1', 'Close_diff_2','Label']]
    #df = df[['High','Low','Open','Close','Close_diff','momentum_stoch_rsi','others_dlr','volatility_kcp','volume_vpt','Label']]

    df.dropna(inplace=True)

    df_ = df[df.index>=datetime.datetime(2022,5,1)]
    df_ = df_[t_limit:]
    df = df[df.index<datetime.datetime(2022,5,1)]

    X = df[df.columns[:-1]]
    y = df.Label
    X_ = df_[df_.columns[:-1]]
    y_ = df_.Label
    forest,X_train,y_train,X_test,y_test,list_features,forest_importances = train_forest_model(X,y,30)

    #clf = DecisionTreeClassifier(max_depth=3)
    #clf.fit(X_train, y_train)
    #tree.plot_tree(clf)
    #X = df[list_features]
    #y = df.Label
    #X_ = df_[list_features]
    #y_ = df_.Label
    #forest,X_train,y_train,X_test,y_test,list_features,forest_importances = train_forest_model(X,y,30)

    print('Train')
    y_pred_t = forest.predict(X_train)
    print(accuracy_score(y_train, y_pred_t))
    print(confusion_matrix(y_train, y_pred_t))

    y_pred_t = pred_prob(forest,X_train,p=0.67) #forest.predict(X_test)
    print(accuracy_score(y_train, y_pred_t))
    print(confusion_matrix(y_train, y_pred_t))

    print('Teste')
    y_pred = pred_prob(forest,X_test,p=0.67) #forest.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print('Teste*')
    y_pred_t_ = pred_prob(forest,X_,p=0.67) #forest.predict(X_)
    print(accuracy_score(y_, y_pred_t_))
    print(confusion_matrix(y_, y_pred_t_))

    df_['Label_pred_'] = y_pred_t_
    plt_lb.plot_label(df_['Close'],df_['Label_pred_'])
    plt_lb.plot_label(df_['Close'],df_['Label'])

    fast_ma = vbt.MA.run(df_['Close'], 10, short_name='fast')
    slow_ma = vbt.MA.run(df_['Close'], 20, short_name='slow')
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('MA:')
    print(pf.total_return())

    df_['entries'] = False
    df_.loc[df_.Label==1,'entries'] = True
    df_['exits'] = False
    df_.loc[df_.Label==-1,'exits'] = True
    entries = df_['entries']
    exits = df_['exits']
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('Modelo Ideal:')
    print(pf.total_return())

    df_['entries'] = False
    df_.loc[df_.Label_pred_==1,'entries'] = True
    df_['exits'] = False
    df_.loc[df_.Label_pred_==-1,'exits'] = True
    entries = df_['entries']
    exits = df_['exits']
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('Modelo:')
    print(pf.total_return())

    df_['entries'] = False
    df_.loc[df_.Label_pred_==1,'entries'] = True
    entries = df_['entries']
    exits = vbt.OHLCSTX.run(
        entries, 
        df_['Open'], 
        df_['High'], 
        df_['Low'], 
        df_['Close'], 
        tp_stop=0.05,
        sl_stop=0.05
    ).exits
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('Modelo_2:')
    print(pf.total_return())

    df_['entries'] = False
    df_.loc[df_.Label_pred_==1,'entries'] = True
    entries = df_['entries']
    df_['exits_model'] = False
    df_.loc[df_.Label_pred_==-1,'exits_model'] = True
    df_['exits_stop'] = vbt.OHLCSTX.run(
                            entries, 
                            df_['Open'], 
                            df_['High'], 
                            df_['Low'], 
                            df_['Close'], 
                            sl_stop=0.05,
                            stop_type=None, 
                            stop_price=None
                        ).exits
    exits = df_[['exits_model','exits_stop']].any(axis=1)
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('Modelo_3:')
    print(pf.total_return())

    entries[:] = True
    entries[0] = True
    exits[:] = False
    pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
    print('BandH:')
    print(pf.total_return())

    return

for df in list_df:
    test(df)

