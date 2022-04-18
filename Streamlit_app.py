import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_market_data():

    tickers_arg = st.sidebar.text_input("What is the ticker? ")
    st.write("Ticker:", tickers_arg)
    period_arg = st.sidebar.text_input("What is the period? ")
    st.write("Period:", period_arg)
    interval_arg = st.sidebar.text_input("What is the interval? ")
    st.write("Interval:", interval_arg)

    st.write("Getting data from "+tickers_arg)

    continue_if = (tickers_arg!="") and (period_arg!="") and (interval_arg!="")

    if continue_if:
        df = yf.download(tickers=tickers_arg, peiod=period_arg, interval=interval_arg)
        st.line_chart(df[['Open','High','Low','Close']])
        return df

def main():

    data = None
    data = get_market_data()

    if data is not None:
        return None
        #ESCREVER AQUI TODA A ANALISE

main()