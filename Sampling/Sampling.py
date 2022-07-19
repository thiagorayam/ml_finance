import pandas as pd

def volume_bar(df,v_size = 600):    
    df_v = df.assign(V_index=lambda row: df['VOLUME'].cumsum()//v_size).reset_index()
    df_v = df_v.groupby(['V_index']).agg({'DATE': 'max',
                                            'OPEN' : 'first',
                                            'CLOSE' : 'last',
                                            'HIGHT' : 'max',
                                            'LOW' : 'min',
                                            'VOLUME' : 'sum'})
    df_v = df_v.sort_values('DATE')
    df_v = df_v.set_index('DATE')
    return df_v

def mpNumCoEvents(closeIdx,t1):
    t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    count=pd.Series(0,index=closeIdx)
    for tIn,tOut in t1.iteritems():count.loc[tIn:tOut]+=1.
    return count

def mpSampleTW(t1,numCoEvents):
# Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=t1.index)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght