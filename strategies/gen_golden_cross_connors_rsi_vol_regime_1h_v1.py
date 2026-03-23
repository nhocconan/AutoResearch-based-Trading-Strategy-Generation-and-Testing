#!/usr/bin/env python3
"""Auto-generated: golden_cross trend + connors_rsi entry + vol_regime regime on 1h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_golden_cross_connors_rsi_vol_regime_1h_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    close_s = pd.Series(close)

    # ATR for stoploss
    _tr = np.zeros(n)
    for i in range(1, n): _tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = pd.Series(_tr).rolling(14, min_periods=14).mean().values

    # TREND indicator

    sma50 = close_s.rolling(50, min_periods=50).mean().values
    sma200 = close_s.rolling(200, min_periods=200).mean().values
    trend = np.where(sma50 > sma200, 1.0, np.where(sma50 < sma200, -1.0, 0.0))

    # ENTRY filter

    # Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    _d = np.diff(close, prepend=close[0])
    _g = np.where(_d>0,_d,0); _l = np.where(_d<0,-_d,0)
    _rsi3 = 100 - 100/(1+np.where(pd.Series(_l).ewm(span=3,min_periods=3,adjust=False).mean().values>0, pd.Series(_g).ewm(span=3,min_periods=3,adjust=False).mean().values/pd.Series(_l).ewm(span=3,min_periods=3,adjust=False).mean().values, 100))
    _streak = np.zeros(n)
    for i in range(1,n): _streak[i] = (_streak[i-1]+1 if close[i]>close[i-1] else (_streak[i-1]-1 if close[i]<close[i-1] else 0))
    _streak_rsi = 100 - 100/(1+np.where(pd.Series(np.where(np.diff(_streak,prepend=0)<0,-np.diff(_streak,prepend=0),0)).ewm(span=2,min_periods=2,adjust=False).mean().values>0, pd.Series(np.where(np.diff(_streak,prepend=0)>0,np.diff(_streak,prepend=0),0)).ewm(span=2,min_periods=2,adjust=False).mean().values/pd.Series(np.where(np.diff(_streak,prepend=0)<0,-np.diff(_streak,prepend=0),0)).ewm(span=2,min_periods=2,adjust=False).mean().values, 100))
    _pct_rank = pd.Series(close.astype(float)).rolling(100,min_periods=50).rank(pct=True).values * 100
    _crsi = (_rsi3 + _streak_rsi + np.nan_to_num(_pct_rank, nan=50)) / 3
    entry_ok_long = np.array([_crsi[i] < 15 for i in range(n)])
    entry_ok_short = np.array([_crsi[i] > 75 for i in range(n)])

    # REGIME filter

    _vol_tr = np.zeros(n)
    for i in range(1, n): _vol_tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _vol_atr = pd.Series(_vol_tr).rolling(14, min_periods=14).mean().values
    _vol_pct = np.where(close > 0, _vol_atr / close, 0)
    _vol_pct_median = pd.Series(_vol_pct).rolling(100, min_periods=50).median().values
    regime_ok = np.array([not np.isnan(_vol_pct_median[i]) and _vol_pct[i] < _vol_pct_median[i] * 1.5 for i in range(n)])

    signals = np.zeros(n)
    SIZE = 0.25
    entry_price = 0.0
    in_trade = 0

    for i in range(100, n):
        if np.isnan(atr[i]) or atr[i] == 0: continue

        # Manage position
        if in_trade != 0:
            if in_trade == 1 and close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == -1 and close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == 1 and trend[i] < 0:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == -1 and trend[i] > 0:
                signals[i] = 0.0; in_trade = 0; continue
            signals[i] = SIZE * in_trade; continue

        if not regime_ok[i]: signals[i] = 0.0; continue

        if trend[i] > 0 and entry_ok_long[i]:
            signals[i] = SIZE; entry_price = close[i]; in_trade = 1
        elif trend[i] < 0 and entry_ok_short[i]:
            signals[i] = -SIZE; entry_price = close[i]; in_trade = -1
        else:
            signals[i] = 0.0

    return signals
