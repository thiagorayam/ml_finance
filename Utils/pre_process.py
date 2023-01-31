import numpy as np
import pandas as pd

def buy_sell_position__(bs_list):
    
    sit_list = np.zeros(len(bs_list))
    bs_list_final = np.zeros(len(bs_list))

    first_buy = np.where(bs_list == 1)[0][0]
    sit_list[first_buy:] = 1
    bs_list_final[first_buy] = 1

    for idx, x in np.ndenumerate(bs_list):
        index = idx[0]
        index_ = max(idx[0] - 1,0)

        if abs(bs_list[index]-sit_list[index_]) == 2:
            sit_list[index:] = bs_list[index]
            bs_list_final[index] = bs_list[index]
    
    return bs_list_final

def buy_sell_position_(bs_list):
    
    sit_list = np.zeros(len(bs_list))
    bs_list_final = np.zeros(len(bs_list))

    first_buy = np.where(bs_list == 1)[0][0]
    sit_list[first_buy:] = 1
    bs_list_final[first_buy] = 1

    for idx, x in np.ndenumerate(bs_list):
        index = idx[0]
        index_ = max(idx[0] - 1,0)

        if abs(bs_list[index]-sit_list[index_]) == 2:
            sit_list[index:] = bs_list[index]
            bs_list_final[index] = bs_list[index]
    
    return bs_list_final

def check_buy_group(signal):

    i=0

    buy_ = False
    sell_ = False

    group = np.zeros(len(signal))

    for idx, x in np.ndenumerate(signal):

        index = idx[0]
        signal_atual = signal[index]
        group[index] = i

        buy = signal_atual==1
        sell = signal_atual==-1

        if buy:
            buy_ = True

        if sell:
             sell_ = True

        if buy_ and sell_:
            if buy:
                i = i+1
                group[index] = i
                buy_ = False
                sell_ = False
        
    return group
           
def buy_sell_position(index_df,signal,close,limit=0.05):

    close_ = close.copy()
    signal_ = signal.copy()

    sit_list = np.zeros(len(signal_))
    bs_list_final = np.zeros(len(signal_))
    close_list = np.zeros(len(close_))

    first_buy = np.where(signal_ == 1)[0][0]
    sit_list[first_buy:] = 1
    bs_list_final[first_buy] = 1
    close_list[:] = close_[first_buy]
    last_stop_loss = False

    for idx in np.arange(len(signal_)):

        index = idx
        index_ = max(idx - 1,0)

        up_lim_cond = ((close_[index]-close_list[index_])/close_[index] >= 2*limit)
        low_lim_cond = ((close_[index]-close_list[index_])/close_[index] <= -limit)

        lim_cond = up_lim_cond or low_lim_cond

        bought_cond = sit_list[index_] == 1

        stop_loss = lim_cond and bought_cond

        change_cond = (abs(signal_[index]-sit_list[index_]) == 2) and not last_stop_loss

        if change_cond | stop_loss:
                        
            close_list[index:] = close_[index]

            if stop_loss:
                last_stop_loss = True
                sit_list[index:] = -1
                bs_list_final[index] = -1
            else:
                sit_list[index:] = signal_[index]
                bs_list_final[index] = signal_[index]

        if  signal_[index] == -1  and sit_list[index_] == -1 and last_stop_loss:
            last_stop_loss = False                      

    return pd.Series(data=bs_list_final,index=index_df)

