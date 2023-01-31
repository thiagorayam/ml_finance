import yfinance as yf
import datetime
import pandas as pd
import numpy as np

from Labeling import get_label
from PlotFunctions import plot

start_date = "2006-08-01"
end_date = "2023-01-01"

stock_index =  yf.download(["ITUB4.SA"],start_date,end_date)

window = 7
labeling = get_label.Labeling(stock_index['Close'])
stock_index['Label'] = labeling.min_max(window, len(stock_index['Close']), 2).fillna(0)           

plot.plot_label(stock_index['Close'][0:20],stock_index['Label'][0:20])
