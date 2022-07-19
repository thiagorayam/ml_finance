import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_candlesticks(df):
	fig = go.Figure(data = [go.Candlestick(x=df.DATE, open = df.OPEN, high = df.HIGHT, low = df.LOW, close = df.CLOSE)])
	fig.show()
	return "figura plotada"