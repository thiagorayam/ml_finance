import yfinance as yf
import pandas as pd

pd.options.mode.chained_assignment = None


class StockList():
    
    def __init__(self,
                list_name =  ['PETR4','VALE3','ITUB4',
                                    'BBDC4','ABEV3','BOVA11'],
                start_date = "2006-08-01",
                end_date = "2023-01-01"
                  ):
        self.list_name = list(list_name)
        self.list = {}
        self.start_date = start_date
        self.end_date = end_date

    def download(self):

        for name in self.list_name:
            stock =  yf.download(
                                [name+".SA"],
                                 self.start_date,
                                 self.end_date)
            stock.name = name
            self.list[stock.name] = stock
