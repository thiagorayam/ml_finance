import matplotlib.pyplot as plt

def plot_label(close,label,figsize=(20,10)):
    
    plt.figure(figsize = figsize)
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