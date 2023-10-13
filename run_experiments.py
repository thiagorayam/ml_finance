import utils.data_pipeline as pipe
from extraction.stock import StockList

STOCK_LIST =  ['PETR4', 'VALE3', 'ITUB4',
               'BBDC4', 'ABEV3', 'BOVA11']
START = "2006-08-01"
END = "2023-08-01"
list_years = [2017,2018,2019,2020,2021,2022,2023]

#for model_name in ['xgb']:
model_name = 'xgb'

file_name = model_name +'_result_3.joblib'
            
#Extract
stock = StockList(STOCK_LIST,START,END)
stock.download()

#Train/Predict
stock = pipe.train_predict(list_years,stock,model_name)

#save
stock.save(file_name)