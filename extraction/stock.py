import yfinance as yf
import joblib
class StockList():
    
    def __init__(self,
                list_name =  ['PETR4','VALE3','ITUB4',
                            'BBDC4','ABEV3','BOVA11'],
                start_date = "2006-08-01",
                end_date = "2023-01-01"
                  ):
        self.list_name = list(list_name)
        self.list = {}
        self.backtest = {}
        self.start_date = start_date
        self.end_date = end_date

    def download(self):

        for name in self.list_name:
            stock =  yf.download(
                                [name+".SA"],
                                 self.start_date,
                                 self.end_date)
            stock = stock.reset_index()
            stock.columns = stock.columns.str.lower().str.replace(' ', '_')
            self.list[name] = stock

    def save(self, file_name):
        self.backtest = {}
        joblib.dump(self,file_name)

    @classmethod
    def load(self, file_name):
        return joblib.load(file_name)
