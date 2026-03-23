#!/usr/bin/env python3
"""Auto-generated: darvas_box trend + connors_rsi entry + choppiness regime on 15m"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_darvas_box_connors_rsi_choppiness_15m_v1"
timeframe = "15m"
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

    _box_high = np.zeros(n); _box_low = np.zeros(n)
    _box_high[0] = high[0]; _box_low[0] = low[0]
    _in_box = True; _box_start = 0
    trend = np.zeros(n)
    for i in range(1, n):
        if _in_box:
            if high[i] > _box_high[_box_start]:
                _box_high[i] = high[i]; _box_start = i; _box_low[i] = low[i]
            elif low[i] < _box_low[_box_start]:
                _box_low[i] = low[i]
            else:
                _box_high[i] = _box_high[i-1]; _box_low[i] = _box_low[i-1]
            if i - _box_start > 10:
                _in_box = False
        else:
            _box_high[i] = _box_high[i-1]; _box_low[i] = _box_low[i-1]
            if close[i] > _box_high[i]: trend[i] = 1.0; _in_box = True; _box_start = i; _box_high[i] = high[i]; _box_low[i] = low[i]
            elif close[i] < _box_low[i]: trend[i] = -1.0; _in_box = True; _box_start = i; _box_high[i] = high[i]; _box_low[i] = low[i]
            else: trend[i] = trend[i-1]

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

    _tr = np.zeros(n)
    for i in range(1, n): _tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _atr_sum = pd.Series(_tr).rolling(14, min_periods=14).sum().values
    _hh = pd.Series(high).rolling(14, min_periods=14).max().values
    _ll = pd.Series(low).rolling(14, min_periods=14).min().values
    _range = _hh - _ll
    chop = np.where(_range > 0, 100 * np.log10(_atr_sum / _range) / np.log10(14), 50)
    regime_ok = np.array([chop[i] < 45 for i in range(n)])

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
