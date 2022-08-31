import numpy as np
import pandas as pd

class Labelling:

    def __init__(self, value):
        self.value = value

    # Min max
    def min_max(self, window, total):
        
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

            win_start = int((win_end + win_start)/2)
            win_end = win_start + window
            count_row = win_end

        return label

    # Triple barrier method
    def tbm(self, pt, sl, limit):

        label = pd.Series(index=self.value.index,dtype='int8').fillna(0)

        for i in range(len(self.value)):

            cum_return = self.value[i:i+limit].diff().fillna(0).cumsum()/self.value[i]
            index_touch = (((cum_return)>pt)|((cum_return)<-sl)).idxmax()

            if cum_return[index_touch]>0:
                label.iloc[i]=1

            elif cum_return[index_touch]<0:
                label.iloc[i]=-1
        
        return label