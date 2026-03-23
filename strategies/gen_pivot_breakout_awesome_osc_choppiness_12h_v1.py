#!/usr/bin/env python3
"""Auto-generated: pivot_breakout trend + awesome_osc entry + choppiness regime on 12h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_pivot_breakout_awesome_osc_choppiness_12h_v1"
timeframe = "12h"
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

    # Daily pivot points
    _prev_h = pd.Series(high).shift(1).rolling(6, min_periods=6).max().values
    _prev_l = pd.Series(low).shift(1).rolling(6, min_periods=6).min().values
    _prev_c = pd.Series(close).shift(1).values
    _pivot = (_prev_h + _prev_l + _prev_c) / 3
    _r1 = 2 * _pivot - _prev_l
    _s1 = 2 * _pivot - _prev_h
    trend = np.zeros(n)
    for i in range(10, n):
        if not np.isnan(_r1[i]):
            if close[i] > _r1[i]: trend[i] = 1.0
            elif close[i] < _s1[i]: trend[i] = -1.0
            else: trend[i] = trend[i-1]

    # ENTRY filter

    _ao_fast = pd.Series((high+low)/2).rolling(5,min_periods=5).mean().values
    _ao_slow = pd.Series((high+low)/2).rolling(34,min_periods=34).mean().values
    _ao = _ao_fast - _ao_slow
    entry_ok_long = np.array([_ao[i]>0 and (i<1 or _ao[i]>_ao[i-1]) for i in range(n)])
    entry_ok_short = np.array([_ao[i]<0 and (i<1 or _ao[i]<_ao[i-1]) for i in range(n)])

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
