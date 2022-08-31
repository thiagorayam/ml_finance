import numpy as np
import pandas as pd

class Indicator:

    def __init__(self, open, high, low, close):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
    
    # Money flow multiplier
    def mfm(self):
        return ((self.close - self.low)- (self.high - self.close))/(self.high - self.low)

    