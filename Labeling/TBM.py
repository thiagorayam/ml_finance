import pandas as pd
import numpy as np

def getTEvents(gRaw, h):
  tEvents, sPos, sNeg = [], 0, 0
  diff = gRaw.diff()
  for i in diff.index[1:]:
    sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
    if sNeg < -h:
      sNeg = 0;
      tEvents.append(i)
    elif sPos > h:
      sPos = 0;
      tEvents.append(i)
  return pd.DatetimeIndex(tEvents)

def getEvents(close,tEvents,ptSl,trgt,minRet,t1,side):
  # 1) get target
  tEvents = tEvents[1:]
  trgt = trgt[trgt.index.isin(tEvents)]
  trgt = trgt[trgt > minRet]  # minRet
  # 2) get t1 (max holding period)
  if t1 is False: 
    t1_ = pd.Series(pd.NaT, index=trgt.index)
  else:
    t1_ = t1[t1.index.isin(trgt.index)]
  # 3) form events object, apply stop loss on t1
  if side is None:
    side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
  else:
    side_, ptSl_ = side[side.index.isin(trgt.index)], ptSl[:2]
  events = pd.concat([t1_,trgt,side_],axis=1)
  events.columns = ['t1', 'trgt', 'side']
  events.dropna(subset=['trgt'],inplace=True)
  df0 = applyPtSlOnT1(close, events, ptSl)
  events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
  if side is None: events = events.drop('side', axis=1)
  return events

def applyPtSlOnT1(close, events, ptSl):
  # apply stop loss/profit taking, if it takes place before t1 (end of event)
  out = events[['t1']].copy(deep=True)
  if ptSl[0]>0:pt=ptSl[0]*events['trgt']
  else:pt=pd.Series(index=events.index) # NaNs
  if ptSl[1]>0:sl=-ptSl[1]*events['trgt']
  else:sl=pd.Series(index=events.index) # NaNs
  for loc,t1 in events['t1'].fillna(close.index[-1]).iteritems():
    df0=close[loc:t1] # path prices
    df0=(df0/close[loc]-1)*events.at[loc,'side'] # path returns
    out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
    out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
  return out

def getBins(events, close):
  # 1) prices aligned with events
  events_ = events.dropna(subset=['t1'])
  px = events_.index.union(events_['t1'].values).drop_duplicates()
  px = close.reindex(px, method='bfill')
  # 2) create out object
  out = pd.DataFrame(index=events_.index)
  out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
  if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
  out['bin'] = np.sign(out['ret'])
  if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
  return out

def getBins_v2(events, close, t1):
  # 1) prices aligned with events
  events_ = events.dropna(subset=['t1'])
  px = events_.index.union(events_['t1'].values).drop_duplicates()
  px = close.reindex(px, method='bfill')
  # 2) create out object
  out = pd.DataFrame(index=events_.index)
  out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
  if 'side' in events_: out['ret'] *= events_['side']  # meta-labeling
  out['bin'] = np.sign(out['ret'])
  if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
  out.loc[out.index.isin(events_[events_['t1'].isin(t1)].index),'ret'] = 0
  return out

def dropLabels(events, minP=.05):
  # apply weights, drop labels with insufficient examples
  while True:
    df0 = events['bin'].value_counts(normalize=True)
    if df0.min() > minP or df0.shape[0] < 3: break
    events = events[events['bin'] != df0.index[df0.argmin()]]
  return events