import matplotlib.pyplot as plt
import pandas as pd

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