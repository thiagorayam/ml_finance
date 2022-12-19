import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_label(close,label):
    
    plt.figure(figsize = (20,10))
    plt.plot(close,color='k',label='Close')

    # long position
    long_cond = close.index.isin(label[(label==1)].index)
    plt.plot(close[long_cond].index, close[long_cond], '^',
                markersize = 15, color='g', label='buy')

    # short position
    short_cond = close.index.isin(label[(label==-1)].index)
    plt.plot(close[short_cond].index, close[short_cond], 'v',
                markersize = 15, color='r', label='sell')
                
    plt.xlabel('Date', fontsize = 15 )
    plt.legend()
    plt.grid()
    plt.show()

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

def plot_position(df,start,end,s1=None, s2=None):
    
    df = df[start:end]

    plt.figure(figsize = (20,10))
    plt.plot(df['CLOSE'],color='k',label='Close')

    # long position
    plt.plot(df[df['side'] == 1].index,df['CLOSE'][df['side'] == 1],
                '^', markersize = 15, color='g', label='buy')

    # short position
    plt.plot(df[df['side'] == -1].index,df['CLOSE'][df['side'] == -1],
                'v', markersize = 15, color='r', label='sell')
    
    if s1 is not None and s2 is not None:

        s1 = s1[start:end]
        s2 = s2[start:end]

        plt.plot(s1,color='b')
        plt.plot(s2,color='y')

                
    plt.xlabel('Date', fontsize = 15 )
    plt.legend()
    plt.grid()
    plt.show()

def plot_trgt(df,start,end,b1=1,b2=1):
    
    df = df[start:end]

    plt.figure(figsize = (20,10))
    plt.plot(df['CLOSE'],color='k',label='Close')

    plt.plot((df['CLOSE']*(1+df['trgt']*b1)).fillna(0), markersize = 15, color='g')
    plt.plot((df['CLOSE']*(1-df['trgt']*b2)).fillna(0), markersize = 15, color='r')
    
                
    plt.xlabel('Date', fontsize = 15 )
    plt.legend()
    plt.grid()
    plt.show()

def plot_candlesticks(df):
	fig = go.Figure(data = [go.Candlestick(x=df.DATE, open = df.OPEN, high = df.HIGHT, low = df.LOW, close = df.CLOSE)])
	fig.show()
	return "figura plotada"

