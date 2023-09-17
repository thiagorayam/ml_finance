import numpy as np
import pandas as pd
import statsmodels.api as sm1

class Labeling:

    def __init__(self, value):
        self.value = value

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

            left_barrier = self.value[win_start:win_end].iloc[0]
            right_barrier = self.value[win_start:win_end].iloc[-1]

            if max_value > left_barrier and max_value > right_barrier:
                label.loc[idx_max] = -1

            if min_value < left_barrier and min_value < right_barrier:
                label.loc[idx_min] = 1

            win_start = win_start + int((win_end - win_start)/d)
            win_end = win_start + window
            count_row = win_end
            
        return label