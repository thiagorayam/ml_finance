import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def st_test():
    #argumentos de entrada no app
    tickers_arg = st.sidebar.text_input("What is the ticker? ")
    st.write("Ticker:", tickers_arg)
    period_arg = st.sidebar.text_input("What is the period? ")
    st.write("Period:", period_arg)
    interval_arg = st.sidebar.text_input("What is the interval? ")
    st.write("Interval:", interval_arg)

    st.write("Getting data from "+tickers_arg)

    tickers_arg = 'UBER'
    period_arg = '5d'
    interval_arg = '1d'

    df = yf.download(tickers=tickers_arg, peiod=period_arg, interval=interval_arg)

    st.write(df['Open'])

st_test()

