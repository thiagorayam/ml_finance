import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix)


def plot_label(close,label,figsize=(20,10),fontsize=20):

    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize   
    
    plt.figure(figsize = figsize)
    plt.plot(close,color='k',label='Fechamento')

    long_cond = close.index.isin(label[(label==1)].index)
    plt.plot(close[long_cond].index, close[long_cond], '^',
                markersize = fontsize, color='g', label='Compra')

    short_cond = close.index.isin(label[(label==-1)].index)
    plt.plot(close[short_cond].index, close[short_cond], 'v',
                markersize = fontsize, color='r', label='Venda')
                
    plt.ylabel('R$', fontsize = fontsize )            
    plt.xlabel('Data', fontsize = fontsize )
    plt.legend( fontsize = fontsize )
    plt.grid()
    plt.show()

def clf_result(stock,
                labels=None,
                avg_type='weighted',
                plot_matrix=False,
                plot=False,
                model_name=None):
    
    metrics = pd.DataFrame()

    for name in stock.list_name:

        true_labels = stock.list[name].dropna()['label']
        predicted_labels = stock.list[name].dropna()['label_pred']

        accuracy = accuracy_score(true_labels, predicted_labels)
        
        precision = precision_score(true_labels,
                                    predicted_labels,
                                    labels=labels,
                                    average=avg_type)
        
        recall = recall_score(true_labels,
                                predicted_labels,
                                labels=labels,
                                average=avg_type)
        
        f1 = f1_score(true_labels,
                        predicted_labels,
                        labels=labels,
                        average=avg_type)
        
        confusion = confusion_matrix(true_labels,
                                predicted_labels)

        metrics_ = {"Ativo" : [name],
                    "Acurácia": [accuracy],
                    "Precisão": [precision],
                    "Revocação": [recall],
                    "F1 Score": [f1]}
        metrics_ = pd.DataFrame(metrics_)
        metrics = pd.concat([metrics,metrics_])

        if plot:
            print(name+'...')
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            
            if plot_matrix:
                plt.imshow(confusion, interpolation='nearest', cmap='Blues')
                plt.title('Matriz de Confusão')
                plt.colorbar()

                plt.xticks([0, 1, 2], ['-1', '0', '1'])
                plt.yticks([0, 1, 2], ['-1', '0', '1'])
                
                plt.xlabel('Rótulos Previstos')
                plt.ylabel('Rótulos Verdadeiros')

                for i in range(confusion.shape[0]):
                    for j in range(confusion.shape[1]):
                        plt.text(j, i, f'{confusion[i, j]}', ha='center', va='center', color='black')

                plt.savefig(f'confusion_{name}_{model_name}.png')
                plt.clf()

        
    return metrics

def bt_bh_value(stock):

    value = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        pf_bh = vbt.Portfolio.from_holding(data['close'])

        value_ = pf_bh.value()
        value_.name = name
        
        value = pd.concat([value,value_],axis=1)

    return value

def bt_random_value(stock, n_trades=50, seed=42):

    value = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        pf_random = vbt.Portfolio.from_random_signals(data['close'], n=[n_trades],
                                                      seed=seed, freq='D')

        value_ = pf_random.value()
        value_.name = name
        
        value = pd.concat([value,value_],axis=1)

    return value

def bt_mac_value(stock):

    value = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        fast_ma = vbt.MA.run(data['close'], 10, short_name='fast')
        slow_ma = vbt.MA.run(data['close'], 20, short_name='slow')
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)  

        pf_mac = vbt.Portfolio.from_signals(data['close'], entries, exits, freq = 'D')

        value_ = pf_mac.value()
        value_.name = name
        
        value = pd.concat([value,value_],axis=1)

    return value

def bt_value(stock):

    value = pd.DataFrame()

    for name in stock.list_name:

        pf = stock.backtest[name]
        value_ = pf.value()
        value_.name = name
        
        value = pd.concat([value,value_],axis=1)

    return value

def bt_result(stock,
            plot=False):

    metrics = pd.DataFrame()

    for name in stock.list_name:

        pf = stock.backtest[name]
        pf_opt = stock.backtest_opt[name]
        metrics_ = {"Ativo" : [name],
                    "Trades": [pf.trades.count()],
                    "Taxa de acerto": [pf.trades.win_rate()],
                    "Índice de Sharpe": [pf.returns_acc.sharpe_ratio()],
                    "Sortino": [pf.returns_acc.sortino_ratio()],
                    "Retorno": [pf.total_return()],
                    "Profit factor": [pf.trades.profit_factor()],
                    "Expectancy": [pf.trades.expectancy()],
                    #"Retorno*": [pf_opt.total_return()],
                    # "Buy and Hold": [pf.total_benchmark_return()]
                    }
        metrics_ = pd.DataFrame(metrics_)
        metrics = pd.concat([metrics,metrics_])

        if plot:
            print(name+'...')
            print('Total return: {}'.format(pf.total_return()))
            print('Win rate: {}'.format(pf.trades.win_rate()))
            print('Sharpe rate: {}'.format(pf.returns_acc.sharpe_ratio()))
            print('-')

    return metrics

def bt_mac_result(stock, sl_stop=None, tp_stop=None):

    metrics = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        fast_ma = vbt.MA.run(data['close'], 10, short_name='fast')
        slow_ma = vbt.MA.run(data['close'], 20, short_name='slow')
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)  

        pf_mac = vbt.Portfolio.from_signals(data['close'],
                                            entries,
                                            exits,
                                            freq = 'D',
                                            sl_stop=sl_stop,
                                            tp_stop=tp_stop)


        metrics_ = {"Ativo" : [name],
                    "Trades": [pf_mac.trades.count()],
                    "Taxa de acerto": [pf_mac.trades.win_rate()],
                    "Índice de Sharpe": [pf_mac.returns_acc.sharpe_ratio()],
                    "Sortino": [pf_mac.returns_acc.sortino_ratio()],
                    "Retorno": [pf_mac.total_return()],
                    "Profit factor": [pf_mac.trades.profit_factor()],
                    "Expectancy": [pf_mac.trades.expectancy()],
                    # "Buy and Hold": [pf_mac.total_benchmark_return()]
                    }
        metrics_ = pd.DataFrame(metrics_)
        metrics = pd.concat([metrics,metrics_])

    return metrics

def bt_bh_result(stock):

    metrics = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        pf_bh = vbt.Portfolio.from_holding(data['close'], freq = 'D')

        metrics_ = {"Ativo" : [name],
                    "Trades": [pf_bh.trades.count()],
                    "Taxa de acerto": [pf_bh.trades.win_rate()],
                    "Índice de Sharpe": [pf_bh.returns_acc.sharpe_ratio()],
                    "Sortino": [pf_bh.returns_acc.sortino_ratio()],
                    "Retorno": [pf_bh.total_return()],
                    "Profit factor": [pf_bh.trades.profit_factor()],
                    "Expectancy": [pf_bh.trades.expectancy()],
                    #"Buy and Hold": [pf_bh.total_benchmark_return()]
                    }
        metrics_ = pd.DataFrame(metrics_)
        metrics = pd.concat([metrics,metrics_])

    return metrics

def bt_random_result(stock, n_trades=50, seed=42, sl_stop=None, tp_stop=None):

    metrics = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        pf_random = vbt.Portfolio.from_random_signals(data['close'],
                                                      n=[n_trades],
                                                      seed=seed,
                                                      freq='D',
                                                      sl_stop=sl_stop,
                                                      tp_stop=tp_stop)

        metrics_ = {"Ativo" : [name],
                    "Trades": [pf_random.trades.count()],
                    "Taxa de acerto": [pf_random.trades.win_rate()],
                    "Índice de Sharpe": [pf_random.returns_acc.sharpe_ratio()],
                    "Sortino": [pf_random.returns_acc.sortino_ratio()],
                    "Retorno": [pf_random.total_return()],
                    "Profit factor": [pf_random.trades.profit_factor()],
                    "Expectancy": [pf_random.trades.expectancy()],
                    # "Buy and Hold": [pf_random.total_benchmark_return()]
                    }
        metrics_ = pd.DataFrame(metrics_)
        metrics = pd.concat([metrics,metrics_])

    return metrics

def wallet_return(value,plot=True):

    returns = value/value.iloc[0]
    returns.columns = ['RF', 'LGBM', 'XGB', 'MAC', 'Random', 'BH']

    if plot:
        plt.figure(figsize=(18, 8))
        columns_to_plot = returns.columns 
        for col in columns_to_plot:
            plt.plot(returns.index, returns[col], label=col)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.title('Retorno Acumulado', fontsize=20)
        plt.xlabel('Data', fontsize=20)
        plt.ylabel('Retorno', fontsize=20)
        plt.legend(loc='best', fontsize=20)
        plt.grid(True)
        plt.show()

    return returns

def bt_random_trades(stock, n_trades=50, seed=42):

    trades = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        pf_random = vbt.Portfolio.from_random_signals(data['close'], n=[n_trades],
                                                      seed=seed, freq='D')

        trades_ = pd.Series((pf_random.trades.values['pnl']>0).astype(int))
        trades_.name = name
        
        trades = pd.concat([trades,trades_],axis=1)

    cum_win_rate = (trades.cumsum().sum(axis=1,skipna=False)/trades.notna().astype(int).cumsum().sum(axis=1)).round(2)

    return cum_win_rate

def bt_mac_trades(stock):

    trades = pd.DataFrame()

    for name in stock.list_name:

        data = stock.list[name]
        data = data[data.label_pred.notna()]
        data = data.set_index('date')

        fast_ma = vbt.MA.run(data['close'], 10, short_name='fast')
        slow_ma = vbt.MA.run(data['close'], 20, short_name='slow')
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)  

        pf_mac = vbt.Portfolio.from_signals(data['close'], entries, exits, freq = 'D')

        trades_ = pd.Series((pf_mac.trades.values['pnl']>0).astype(int))
        trades_.name = name
        
        trades = pd.concat([trades,trades_],axis=1)

    cum_win_rate = (trades.cumsum().sum(axis=1,skipna=False)/trades.notna().astype(int).cumsum().sum(axis=1)).round(2)

    return cum_win_rate

def bt_trades(stock):

    trades = pd.DataFrame()

    for name in stock.list_name:

        pf = stock.backtest[name]

        trades_ = pd.Series((pf.trades.values['pnl']>0).astype(int))
        trades_.name = name
        
        trades = pd.concat([trades,trades_],axis=1)

    cum_win_rate = (trades.cumsum().sum(axis=1,skipna=False)/trades.notna().astype(int).cumsum().sum(axis=1)).round(2)

    return cum_win_rate

def plot_win_cum_rate(total_trades):

    plt.figure(figsize=(18, 8))
    columns_to_plot = total_trades.columns 
    for col in columns_to_plot:
        plt.plot(total_trades.index, total_trades[col], label=col)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.title('Taxa de Acerto Acumulada', fontsize=20)
    plt.xlabel('Número de Trades', fontsize=20)
    plt.ylabel('Taxa de acerto', fontsize=20)
    plt.legend(loc='best', fontsize=20)
    plt.grid(True)
    plt.show()