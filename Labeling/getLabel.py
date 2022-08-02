import numpy as np
import pandas as pd

def getLabel(close,numDays,trgt,minRet=0.005,maxRet=0.05,barrier=[1,1]):

    t1 = close.index.searchsorted(close.index+pd.Timedelta(days=numDays))
    t1 = t1[t1<close.shape[0]]
    t1 = pd.Series(close.index[t1],index=close.index[:t1.shape[0]])
    t1 = t1[t1.index.isin(trgt.index)]

    trgt.loc[trgt<minRet] = minRet
    trgt.loc[trgt>maxRet] = maxRet
    
    upper_barrier = barrier[0]*trgt
    lower_barrier = -barrier[1]*trgt

    out = pd.DataFrame(index=close.index)
   
    for start,end in t1.iteritems():
        
        close_aux = close[start:end]
        return_aux = (close_aux/close_aux[start]-1)

        upper_touch = return_aux[return_aux>upper_barrier[start]].index.min()
        lower_touch = return_aux[return_aux<lower_barrier[start]].index.min()
       
        try:
            out.loc[start,'return'] = return_aux[min([lower_touch,upper_touch])]
        except: 
            out.loc[start,'return'] = 0

    return out