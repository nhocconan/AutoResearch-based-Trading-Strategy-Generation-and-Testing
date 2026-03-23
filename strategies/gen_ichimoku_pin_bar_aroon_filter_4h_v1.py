#!/usr/bin/env python3
"""Auto-generated: ichimoku trend + pin_bar entry + aroon_filter regime on 4h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_ichimoku_pin_bar_aroon_filter_4h_v1"
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

    tenkan = (pd.Series(high).rolling(20).max().values + pd.Series(low).rolling(20).min().values) / 2
    kijun = (pd.Series(high).rolling(60).max().values + pd.Series(low).rolling(60).min().values) / 2
    trend = np.zeros(n)
    for i in range(60, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            trend[i] = 1.0 if tenkan[i] > kijun[i] and close[i] > kijun[i] else (-1.0 if tenkan[i] < kijun[i] and close[i] < kijun[i] else 0.0)

    # ENTRY filter

    _body = np.abs(close - prices["open"].values) if "open" in prices.columns else np.ones(n)
    _range = high - low
    _upper_wick = high - np.maximum(close, prices["open"].values if "open" in prices.columns else close)
    _lower_wick = np.minimum(close, prices["open"].values if "open" in prices.columns else close) - low
    entry_ok_long = np.array([_lower_wick[i] > 2*_body[i] and _range[i]>0 for i in range(n)])
    entry_ok_short = np.array([_upper_wick[i] > 2*_body[i] and _range[i]>0 for i in range(n)])

    # REGIME filter

    aroon_up = np.zeros(n); aroon_dn = np.zeros(n)
    for i in range(25, n):
        hh_idx = i - 25 + np.argmax(high[i-25:i])
        ll_idx = i - 25 + np.argmin(low[i-25:i])
        aroon_up[i] = (25 - (i - hh_idx)) / 25 * 100
        aroon_dn[i] = (25 - (i - ll_idx)) / 25 * 100
    regime_ok = np.array([abs(aroon_up[i] - aroon_dn[i]) > 30 for i in range(n)])

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
