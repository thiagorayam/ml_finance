import numpy as np
import pandas as pd
import statsmodels.api as sm1

class Labelling:

    def __init__(self, value):
        self.value = value

    def getDailyVol(self, span0=100):
        df0 = self.value.index.searchsorted(self.value.index - pd.Timedelta(days=1))
        df0 = df0[df0 > 0]
        df0 = pd.Series(self.value.index[df0 - 1],
                        index = self.value.index[self.value.shape[0]
                        - df0.shape[0]:])
        df0 = self.value.loc[df0.index] / self.value.loc[df0.values].values - 1  
        df0 = df0.ewm(span=span0).std()
        self.daily_vol = df0.fillna(0)
        return df0

    # Min max
    def min_max(self, window, total, d):
        
        label = pd.Series(index=self.value.index,dtype='int8').fillna(0)

        win_start = 0
        win_end = win_start + window
        count_row = 0

        while count_row <= total:
        
            min_value = min(self.value[win_start:win_end])
            max_value = max(self.value[win_start:win_end])
            idx_min = self.value[win_start:win_end].idxmin()
            idx_max = self.value[win_start:win_end].idxmax()
            
            label.loc[idx_min] = 1
            label.loc[idx_max] = -1

            win_start = int((win_end + win_start)/d)
            win_end = win_start + window
            count_row = win_end

        return label

    # Triple barrier method
    def tbm(self, pt, sl, limit, f=0):

        label = pd.Series(index=self.value.index,dtype='int8').fillna(0)

        for i in range(len(self.value)):

            cum_return = self.value[i:i+limit].diff().fillna(0).cumsum()/self.value[i]
            try:
                daily_vol = self.daily_vol.loc[self.daily_vol.index==self.value.index[i]][0]
                daily_vol = abs(daily_vol)
            except:
                print('Erro daily_vol')
          
            if f>0:
                factor = f
            else:
                factor = daily_vol

            index_touch = (((cum_return)>pt*factor)|((cum_return)<-sl*factor)).idxmax()

            if cum_return[index_touch]>0:
                label.iloc[i]=1

            elif cum_return[index_touch]<0:
                label.iloc[i]=-1
        
        return label


    def getBinsFromTrend(self,span=20):

        def tValLinR(close):
            # tValue from a linear trend
            x=np.ones((close.shape[0],2))
            x[:,1]=np.arange(close.shape[0])
            ols=sm1.OLS(close,x).fit()
            return ols.tvalues[1]

        molecule = self.value.index
        close = self.value

        """
        Derive labels from the sign of t-value of linear trend
        Output includes:
        - t1: End time for the identified trend
        - tVal: t-value associated with the estimated trend coefficient
        - bin: Sign of the trend
        """
        out=pd.DataFrame(index=molecule,columns=['t1','tVal','bin'])
        hrzns=range(span)
        for dt0 in molecule:    
            df0 = pd.Series()
            iloc0 = close.index.get_loc(dt0)
            if iloc0+max(hrzns)>close.shape[0]-1:continue 
            for hrzn in hrzns:
                dt1 = close.index[iloc0+hrzn]
                df1 = close.loc[dt0:dt1]
                df0.loc[dt1]=tValLinR(df1.values)
            dt1=df0.replace([-np.inf,np.inf,np.nan],0).abs().idxmax()
            out.loc[dt0,['t1','tVal','bin']]=df0.index[-1],df0[dt1],np.sign(df0[dt1]) # prevent leakage
        out['t1']=pd.to_datetime(out['t1'])
        out['bin']=pd.to_numeric(out['bin'],downcast='signed')
        return out['bin']