import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import Labeling.get_label as lb
import FractDiff.fractdiff as fdiff
import vectorbt as vbt
import Utils.pre_process as pp
import ta
#https://technical-analysis-library-in-python.readthedocs.io/en/latest/
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,average_precision_score,recall_score,f1_score,classification_report

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None

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
BOVA11 =  yf.download(["BOVA11.SA"],start_date,end_date)
BOVA11.name = 'BOVA11'

list_stock = [PETR4,VALE3,ITUB4,BBDC4,ABEV3,BOVA11]
list_year = [2017,2018,2019,2020,2021,2022]

df_result = pd.DataFrame()
df_result_model = pd.DataFrame()

for j in range(100):
    print(j)

    df_dict = {}

    for stock_index in list_stock:

        df__ = pd.DataFrame()
        for year in list_year:

            print(stock_index.name)
            print(year)

            first_date = stock_index.index>=datetime.datetime(year-10,1,1)
            last_date = stock_index.index<datetime.datetime(year+1,1,1)

            df = stock_index[first_date & last_date]
            df.name = stock_index.name

            adf_test = fdiff.MemoryVsCorr(df['Close'], [0,1], 20, 10, 'Close')
            d=adf_test[adf_test['adf']<=adf_test['1%'].min()]['order'].min()

            n=4

            df['Close_'] = fdiff.ts_differencing(df[['Close']],d,10)
            df['Open_'] = fdiff.ts_differencing(df[['Open']],d,10)
            df['High_'] = fdiff.ts_differencing(df[['High']],d,10)
            df['Low_'] = fdiff.ts_differencing(df[['Low']],d,10)

            df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
            df.dropna(axis=1, how='all', inplace=True)

            
            N = n
            df['Close_diff'] = fdiff.ts_differencing(df[['Close']],d,10)
            close_diff_list = []
            for x in range(1, N + 1):
                df['Close_diff_'+str(x)] = df['Close_diff'].shift(x)
                close_diff_list = close_diff_list + ['Close_diff_'+str(x)]
            df.dropna(inplace=True)
           
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

            forest = RandomForestClassifier(class_weight='balanced_subsample',
                                            n_estimators=100,
                                            max_depth=9,
                                            min_samples_leaf=6,
                                            bootstrap=True,
                                            random_state=j)
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

            df_['Label_pred'] = y_pred_t_ 

            df__ = pd.concat([df__,df_])

        df_dict[stock_index.name] = df__

    result = pd.DataFrame(columns=['ativo','modelo_otm', 'modelo_otm2','modelo', 'modelo2', 'bandhold','ma'])
    result_model = pd.DataFrame(columns=['ativo','accuracy','precision','recall','f1'])

    i = 0
    for stock_name in df_dict.keys():
        print(stock_name)

        #stock_name = 'PETR4'

        df__ = df_dict[stock_name]
        result.loc[i,'ativo'] = stock_name
        result_model.loc[i,'ativo'] = stock_name

        df_ = df__.drop_duplicates()
        df_['Label_pred_'] = df_['Label_pred'].shift(+1).fillna(0)

        df_['Signal'] = pp.buy_sell_position(df_.index,np.array(df_['Label']),df_['Close'],0.1)
        entries = df_['Signal']==1
        exits = df_['Signal']==-1
        pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
        print('Modelo:')
        print(pf.total_return())
        pf.plot()
        result.loc[i,'modelo_otm'] = pf.total_return()

        df_['Signal'] = pp.buy_sell_position_(np.array(df_['Label']))
        entries = df_['Signal']==1
        exits = df_['Signal']==-1
        pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
        print('Modelo:')
        print(pf.total_return())
        pf.plot()
        result.loc[i,'modelo_otm2'] = pf.total_return()

        df_['Signal'] = pp.buy_sell_position(df_.index,np.array(df_['Label_pred_']),df_['Close'],0.1)
        entries = df_['Signal']==1
        exits = df_['Signal']==-1
        pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
        print('Modelo:')
        print(pf.total_return())
        pf.plot()
        result.loc[i,'modelo'] = pf.total_return()

        df_['Signal_'] = pp.buy_sell_position_(np.array(df_['Label_pred_']))
        entries = df_['Signal_']==1
        exits = df_['Signal_']==-1
        pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
        print('Modelo2:')
        print(pf.total_return())
        pf.plot()
        result.loc[i,'modelo2'] = pf.total_return()

        entries[1:] = False
        entries[0] = True
        exits[:] = False
        pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
        print('BandH:')
        print(pf.total_return())
        pf.plot()
        result.loc[i,'bandhold'] = pf.total_return()

        fast_ma = vbt.MA.run(df_['Close'], 10, short_name='fast')
        slow_ma = vbt.MA.run(df_['Close'], 20, short_name='slow')
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        pf = vbt.Portfolio.from_signals(df_['Close'], entries, exits)
        print('MA:')
        print(pf.total_return())
        pf.plot()
        result.loc[i,'ma'] = pf.total_return()

        result_model.loc[i,'accuracy'] = accuracy_score(df_['Label'], df_['Label_pred'])
        result_model.loc[i,'precision'] = precision_score(df_['Label'], df_['Label_pred'],average='weighted')
        result_model.loc[i,'recall'] = recall_score(df_['Label'], df_['Label_pred'],average='weighted')
        result_model.loc[i,'f1'] = f1_score(df_['Label'], df_['Label_pred'],average='weighted')

        i = i + 1


    result['sim_id'] = j
    df_result = pd.concat([df_result,result])
    
    result_model['sim_id'] = j
    df_result_model = pd.concat([df_result_model,result_model])
    print(j+4)
    print('Resultado.......')
    print(df_result)

df_result.to_clipboard()
df_result_model.to_clipboard()
