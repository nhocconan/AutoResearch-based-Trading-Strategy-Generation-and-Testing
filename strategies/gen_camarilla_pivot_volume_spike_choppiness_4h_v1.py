#!/usr/bin/env python3
"""Auto-generated: camarilla_pivot trend + volume_spike entry + choppiness regime on 4h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_camarilla_pivot_volume_spike_choppiness_4h_v1"
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

    # Camarilla Pivot Points (Pivot Boss method)
    _prev_h = pd.Series(high).shift(1).values
    _prev_l = pd.Series(low).shift(1).values
    _prev_c = pd.Series(close).shift(1).values
    _range = _prev_h - _prev_l
    _r3 = _prev_c + _range * 1.1 / 4
    _r4 = _prev_c + _range * 1.1 / 2
    _s3 = _prev_c - _range * 1.1 / 4
    _s4 = _prev_c - _range * 1.1 / 2
    trend = np.zeros(n)
    for i in range(2, n):
        if np.isnan(_r4[i]): continue
        if close[i] > _r4[i]: trend[i] = 1.0  # breakout above R4
        elif close[i] < _s4[i]: trend[i] = -1.0  # breakdown below S4
        elif _s3[i] < close[i] < _r3[i]: trend[i] = 0.0  # inside range = flat
        else: trend[i] = trend[i-1]

    # ENTRY filter

    vol_avg = pd.Series(volume).rolling(20, min_periods=20).mean().values
    vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 1.0)
    entry_ok_long = np.array([vol_ratio[i] > 1.2 for i in range(n)])
    entry_ok_short = entry_ok_long.copy()

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
