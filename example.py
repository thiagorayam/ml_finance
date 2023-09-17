import pandas as pd
import numpy as np
import datetime

from Extraction.stock import StockList
from Labeling import get_label
from PlotFunctions import plot

STOCK_LIST =  ['PETR4','VALE3','ITUB4','BBDC4','ABEV3','BOVA11']
START = "2006-08-01"
END = "2023-01-01"

#Extraction
stock = StockList(STOCK_LIST,START,END)
stock.download()
