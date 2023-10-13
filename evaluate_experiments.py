import pandas as pd
import utils.data_pipeline as pipe
import utils.utils as ut
from extraction.stock import StockList

sl_stop=0.15
tp_stop=0.3

metrics_clf = pd.DataFrame()
metrics_bt = pd.DataFrame()
value = pd.DataFrame()
trades = pd.DataFrame()

for model_name in ['rf','lgbm','xgb']:

    # model_name = 'lgbm'
    file_name = 'experiments/' + model_name +'_result.joblib'

    # load
    stock = StockList()
    stock = stock.load(file_name)

    # clf Result
    metrics_clf_ = ut.clf_result(stock,labels=None,
                              avg_type='macro',
                              plot=False,
                              plot_matrix=False,
                              model_name=model_name)
    metrics_clf_['Modelo'] = model_name
    metrics_clf = pd.concat([metrics_clf,metrics_clf_])

    #Backtesting
    stock = pipe.backtesting(stock,
                            sl_stop=sl_stop,
                            tp_stop=tp_stop)

    # bt Result
    metrics_bt_ = ut.bt_result(stock, plot=False)
    metrics_bt_['Modelo'] = model_name
    metrics_bt = pd.concat([metrics_bt,metrics_bt_])

    #bt Value
    value_ = ut.bt_value(stock)
    value_ = value_.fillna(method='ffill')
    value_ = value_.sum(axis=1)
    value_.name = model_name
    value = pd.concat([value,value_],axis=1)

    #bt trades
    trades_ = ut.bt_trades(stock)
    trades_.name = model_name
    trades = pd.concat([trades,trades_],axis=1)

# Classifier
metrics_clf = metrics_clf.pivot(columns=['Modelo'],index=['Ativo'])

# Backtesting
metrics_bt_mac = ut.bt_mac_result(stock,
                                sl_stop=sl_stop,
                                tp_stop=tp_stop)
metrics_bt_mac['Modelo'] = 'MAC'
metrics_bt_random = ut.bt_random_result(stock,
                                        n_trades=50,
                                        seed=42,
                                        sl_stop=sl_stop,
                                        tp_stop=tp_stop)
metrics_bt_random['Modelo'] = 'Random'
metrics_bt_bh = ut.bt_bh_result(stock)
metrics_bt_bh['Modelo'] = 'BH'
metrics_bt_aux = pd.concat([metrics_bt_random,metrics_bt_mac,metrics_bt_bh])
metrics_bt = pd.concat([metrics_bt,metrics_bt_aux])
metrics_bt = metrics_bt.pivot(columns=['Modelo'],index=['Ativo'])
metrics_bt['Taxa de acerto'] = metrics_bt['Taxa de acerto'].round(2)
metrics_bt['Índice de Sharpe'] = metrics_bt['Índice de Sharpe'].round(2)
metrics_bt['Sortino'] = metrics_bt['Sortino'].round(2)
metrics_bt['Retorno'] = metrics_bt['Retorno'].round(2)
metrics_bt['Profit factor'] = metrics_bt['Profit factor'].round(2)
metrics_bt['Expectancy'] = metrics_bt['Expectancy'].round(2)

# Backtesting value
value_mac = ut.bt_mac_value(stock)
value_mac = value_mac.fillna(method='ffill')
value_mac = value_mac.sum(axis=1)
value_mac.name = 'mac'

value_random = ut.bt_random_value(stock, n_trades=50, seed=42)
value_random = value_random.fillna(method='ffill')
value_random = value_random.sum(axis=1)
value_random.name = 'random'

value_bh = ut.bt_bh_value(stock)
value_bh = value_bh.fillna(method='ffill')
value_bh = value_bh.sum(axis=1)
value_bh.name = 'bh'

total_value = pd.concat([value,value_mac,value_random,value_bh],axis=1)
returns = ut.wallet_return(total_value, plot=True)

# Backtesting cum win_rate
trades_mac = ut.bt_mac_trades(stock)
trades_mac.name = 'mac'

trades_random = ut.bt_random_trades(stock, n_trades=50, seed=42)
trades_random.name = 'random'

total_trades = pd.concat([trades,trades_mac,trades_random],axis=1)
total_trades.index = ((1+total_trades.index)*6) #6 ativos
total_trades.columns = ['RF', 'LGBM', 'XGB', 'MAC', 'Random']
ut.plot_win_cum_rate(total_trades)
