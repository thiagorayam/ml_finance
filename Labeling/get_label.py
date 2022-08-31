import numpy as np
import pandas as pd

class Labelling:

    def __init__(self, value):
        self.value = value
    
    # Money flow multiplier
    def min_max(self, window, total):
        win_start = 0
        win_end = win_start + window
        count_row = 0
        while count_row <= total:
            min_value = min(self.value[win_start:win_end])
            max_value = max(self.value[win_start:win_end])
            index_max + 1 -> -1
            # index_min +1 -> 1
            # else -> 0
            win_start = 