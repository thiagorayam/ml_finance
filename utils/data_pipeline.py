from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from skopt.space import Integer
from skopt import gp_minimize
import vectorbt as vbt
import pandas as pd
import datetime
import ta

from labeling.get_label import Labeling
from fractdiff import fractdiff as fdiff

def get_indicators(data,filter=True):
            
    data =  ta.add_all_ta_features(data,
                                    open="open",
                                    high="high",
                                    low="low",
                                    close="close",
                                    volume="volume",
                                    fillna=True)
    if filter:
                
        ind_list = ['open',
                    'high',
                    'low',
                    'close',
                    'volume',
                    'volume_cmf',
                    'volume_fi',
                    'volume_em',
                    'volume_sma_em',
                    'volume_vpt',
                    'volume_mfi',
                    'volatility_bbw',
                    'volatility_bbp',
                    'volatility_bbhi',
                    'volatility_bbli',
                    'volatility_kcw',
                    'volatility_kcp',
                    'volatility_kchi',
                    'volatility_kcli',
                    'volatility_dcw',
                    'volatility_dcp',
                    'volatility_atr',
                    'volatility_ui',
                    'trend_macd',
                    'trend_macd_signal',
                    'trend_macd_diff',
                    'trend_vortex_ind_pos',
                    'trend_vortex_ind_neg',
                    'trend_vortex_ind_diff',
                    'trend_trix',
                    'trend_mass_index',
                    'trend_dpo',
                    'trend_kst',
                    'trend_kst_sig',
                    'trend_kst_diff',
                    'trend_stc',
                    'trend_adx',
                    'trend_adx_pos',
                    'trend_adx_neg',
                    'trend_cci',
                    'trend_aroon_up',
                    'trend_aroon_down',
                    'trend_aroon_ind',
                    'trend_psar_up_indicator',
                    'trend_psar_down_indicator',
                    'momentum_rsi',
                    'momentum_stoch_rsi',
                    'momentum_stoch_rsi_k',
                    'momentum_stoch_rsi_d',
                    'momentum_tsi',
                    'momentum_uo',
                    'momentum_stoch',
                    'momentum_stoch_signal',
                    'momentum_wr',
                    'momentum_ao',
                    'momentum_roc',
                    'momentum_ppo',
                    'momentum_ppo_signal',
                    'momentum_ppo_hist',
                    'momentum_pvo',
                    'momentum_pvo_signal',
                    'momentum_pvo_hist',
                    'others_dr',
                    'others_dlr',
                    ]

        data = data.loc[:,data.columns.isin(['date']+ind_list)]           

    return data

def get_label(data,type='min_max',params=[]):

    labeling = Labeling(data['close'])
    
    if type == 'min_max':
        data['label'] = labeling.min_max(params[0])

    return data

def transform(data,test_type='5%',lag_cutoff=10,dict_order={}):

    list_series = ['open','high','low','close']
    l = len(dict_order)

    for serie in list_series:
        if l==0:
            adf_test = fdiff.mem_corr(data[serie],[0,1], 20, lag_cutoff, test_type)
            order = adf_test[adf_test['adf']<=
                            adf_test[test_type].min()]['order'].min()
            data[serie] = fdiff.ts_differencing(data[serie],order,lag_cutoff)
            dict_order[serie] = order
        else:
            data[serie] = fdiff.ts_differencing(data[serie],dict_order[serie],lag_cutoff)

    return data, dict_order

def shift_(data, n=4):

    for var in list(set(list(data.columns)) - set(['date','label'])):
        for x in range(1, n + 1):
            data[var+'_'+str(x)] = data[var].shift(x)
    return data

def train_predict(list_years,
                stock,
                model_name='lgbm'):
    
    model_dict = {
            'rf' : RandomForestClassifier(class_weight='balanced'),
            'lgbm' : LGBMClassifier(class_weight='balanced'),
            'xgb' : XGBClassifier()
            }

    for name in list(stock.list.keys()):

        stock.list[name] = get_indicators(stock.list[name],filter=True)
        stock.list[name] = get_label(stock.list[name],'min_max',[7])

        data_oos_result = pd.Series(name='label_pred')

        if model_name == 'xgb':
            stock.list[name]['label'] = stock.list[name]['label'] + 1

        for year in list_years:
                
            first_train = stock.list[name]['date']>=datetime.datetime(year-(5),1,1)
            last_train = stock.list[name]['date']<datetime.datetime(year-(1),1,1)
            first_test = stock.list[name]['date']>=datetime.datetime(year-(1),1,1)
            last_test = stock.list[name]['date']<datetime.datetime(year,1,1)
            first_oos = stock.list[name]['date']>=datetime.datetime(year,1,1)
            last_oos = stock.list[name]['date']<datetime.datetime(year+(1),1,1)

            data = stock.list[name][first_train & last_test]
            data, dict_order = transform(data)
            data = shift_(data)

            data_train = stock.list[name][first_train & last_train]
            data_train, dict_order = transform(data_train, dict_order)
            data_train = shift_(data_train)

            data_test = stock.list[name][first_test & last_test]
            data_test, dict_order = transform(data_test, dict_order)
            data_test = shift_(data_test)

            data_oos = stock.list[name][first_oos & last_oos].set_index('date')
            data_oos, dict_order = transform(data_oos, dict_order)
            data_oos = shift_(data_oos)

            data_train = data_train.dropna()
            X_train = data_train.loc[:,~data_train.columns.isin(
                                        ['date','label','label_pred']
                                        )]
            y_train = data_train['label']

            data_test = data_test.dropna()
            X_test = data_test.loc[:,~data_test.columns.isin(
                                        ['date','label','label_pred']
                                        )]
            y_test = data_test['label']

            #selection
            clf = model_dict[model_name]

            if model_name == 'xgb':
                sample_weights = compute_sample_weight(
                                                class_weight='balanced',
                                                y=y_train
                                                )

                clf.fit(X_train,y_train, sample_weight=sample_weights)
            else:
                clf.fit(X_train,y_train)

            features = X_train.columns
            importances = clf.feature_importances_
            importance_df = pd.DataFrame({'features': features, 'importances': importances})
            importance_df = importance_df.sort_values(by='importances', ascending=False)

            def feature_selection(n_total_features):
                best_features = importance_df.head(n_total_features)
                X_train_ = data_train.loc[:,data_train.columns.isin(best_features.features)]
                X_test_ = data_test.loc[:,data_test.columns.isin(best_features.features)]
                return X_train_,X_test_

            #opt
            param_space = [
                Integer(2, 16, name='max_depth'),
                Integer(50, 200, name='n_estimators'),
                Integer(40, 60, name='n_total_features'),
                ]

            def objective(params):

                params_ = {'max_depth': params[0],
                           'n_estimators': params[1]}
                clf.set_params(**params_)
                X_train_,X_test_ = feature_selection(params[2])

                if model_name == 'xgb':
                    sample_weights = compute_sample_weight(
                                                    class_weight='balanced',
                                                    y=y_train
                                                    )

                    clf.fit(X_train_,y_train, sample_weight=sample_weights)
                else:
                    clf.fit(X_train_,y_train)               
                accuracy = accuracy_score(y_test, clf.predict(X_test_))

                return -accuracy

            result = gp_minimize(objective, param_space,
                                n_calls=50, random_state=0)
            best_params = result.x
            params_ = {'max_depth': best_params[0],
                       'n_estimators': best_params[1]}

            clf = model_dict[model_name]

            clf.set_params(**params_)
            best_features = importance_df.head(best_params[2])

            data = data.dropna()
            X = data.loc[:,data.columns.isin(best_features.features)]
            y = data['label']

            if model_name == 'xgb':
                sample_weights = compute_sample_weight(
                                                class_weight='balanced',
                                                y=y
                                                )

                clf.fit(X,y, sample_weight=sample_weights)
            else:
                clf.fit(X,y)

            data_oos = data_oos.dropna()
            X_oos = data_oos.loc[:,data_oos.columns.isin(best_features.features)]
            X_oos = X_oos[X.columns]
            y_pred_oos = clf.predict(X_oos)

            oos_result = pd.Series(y_pred_oos,
                                    index=X_oos.index,
                                    name='label_pred')
            data_oos_result = pd.concat([data_oos_result,oos_result])

        data_oos_result = data_oos_result.reset_index()
        data_oos_result.columns = ['date','label_pred']
        data_oos_result = data_oos_result.dropna()
        data_oos_result = data_oos_result.drop_duplicates()
        stock.list[name] = stock.list[name].merge(data_oos_result,how='left',on='date')

        if model_name == 'xgb':
            stock.list[name]['label'] = stock.list[name]['label'] - 1
            stock.list[name]['label_pred'] = stock.list[name]['label_pred'] - 1

    return stock

def backtesting(stock, sl_stop = None, tp_stop = None, plot = False):

    stock.backtest = {}
    stock.backtest_opt = {}

    for name in stock.list_name:

        data = stock.list[name][['date',
                                 'open',
                                 'high',
                                 'low',
                                 'close',
                                 'label_pred',
                                 'label']]
        data = data.dropna()
        data = data.set_index('date')
        long = data['label_pred'].shift(1)>0
        short = data['label_pred'].shift(1)<0

        pf = vbt.Portfolio.from_signals(
                                        open = data['open'],
                                        high = data['high'],
                                        low = data['low'],
                                        close = data['close'],
                                        entries = long, 
                                        exits = short,
                                        freq = 'D',
                                        sl_stop = sl_stop,
                                        tp_stop = tp_stop)
        
        stock.backtest[name] = pf

        long_opt = data['label'].shift(1)>0
        short_opt = data['label'].shift(1)<0

        pf_opt = vbt.Portfolio.from_signals(
                                open = data['open'],
                                high = data['high'],
                                low = data['low'],
                                close = data['close'],
                                entries = long_opt, 
                                exits = short_opt,
                                freq = 'D',
                                sl_stop = sl_stop,
                                tp_stop = tp_stop)

        stock.backtest_opt[name] = pf_opt

        if plot:
            print('Total return: {}'.format(pf.total_return()))
            print('Win rate: {}'.format(pf.trades.win_rate()))
            print('Sharpe rate: {}'.format(pf.returns_acc.sharpe_ratio()))
            print('Expectancy: {}'.format(pf.trades.expectancy()))
            print('Profit factor: {}'.format(pf.trades.profit_factor()))
                    
    return stock
