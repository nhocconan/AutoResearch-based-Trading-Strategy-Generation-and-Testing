#!/usr/bin/env python3
"""Auto-generated: darvas_box trend + stochastic entry + none regime on 4h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_darvas_box_stochastic_none_4h_v1"
timeframe = "4h"
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

    low_min = pd.Series(low).rolling(14, min_periods=14).min().values
    high_max = pd.Series(high).rolling(14, min_periods=14).max().values
    stoch_k = np.where(high_max-low_min > 0, (close-low_min)/(high_max-low_min)*100, 50)
    stoch_d = pd.Series(stoch_k).rolling(3, min_periods=3).mean().values
    entry_ok_long = stoch_k < 30
    entry_ok_short = stoch_k > 60

    # REGIME filter
    regime_ok = np.ones(n, dtype=bool)

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
