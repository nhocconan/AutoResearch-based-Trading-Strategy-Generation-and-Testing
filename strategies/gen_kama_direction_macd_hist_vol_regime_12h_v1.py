#!/usr/bin/env python3
"""Auto-generated: kama_direction trend + macd_hist entry + vol_regime regime on 12h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_kama_direction_macd_hist_vol_regime_12h_v1"
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

    kama = np.zeros(n); kama[10] = close[10]
    for i in range(11, n):
        direction_k = abs(close[i] - close[i-10])
        volatility_k = sum(abs(close[j]-close[j-1]) for j in range(i-9, i+1))
        er = direction_k / volatility_k if volatility_k > 0 else 0
        sc = (er * (2/3 - 2/31) + 2/31) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    trend = np.zeros(n)
    for i in range(12, n): trend[i] = 1.0 if close[i] > kama[i] and kama[i] > kama[i-1] else (-1.0 if close[i] < kama[i] and kama[i] < kama[i-1] else 0.0)

    # ENTRY filter

    macd_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    macd_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    macd_line = macd_fast - macd_slow
    macd_signal = pd.Series(macd_line).ewm(span=9, min_periods=9, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    entry_ok_long = np.array([macd_hist[i] > 0 for i in range(n)])
    entry_ok_short = np.array([macd_hist[i] < 0 for i in range(n)])

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
