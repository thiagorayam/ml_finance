import matplotlib.pyplot as plt

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