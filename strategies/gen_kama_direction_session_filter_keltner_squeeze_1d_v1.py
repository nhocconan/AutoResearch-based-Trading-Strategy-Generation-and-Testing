#!/usr/bin/env python3
"""Auto-generated: kama_direction trend + session_filter entry + keltner_squeeze regime on 1d"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_kama_direction_session_filter_keltner_squeeze_1d_v1"
timeframe = "1d"
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

    # Time-of-day seasonality: best crypto returns 20:00-00:00 UTC
    _hours = np.zeros(n)
    if 'open_time' in prices.columns:
        _hours = pd.to_datetime(prices['open_time']).dt.hour.values
    entry_ok_long = np.array([18 <= int(_hours[i]) <= 23 or int(_hours[i]) < 2 for i in range(n)])
    entry_ok_short = entry_ok_long.copy()

    # REGIME filter

    _kc_mid = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    _kc_tr = np.zeros(n)
    for i in range(1, n): _kc_tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _kc_atr = pd.Series(_kc_tr).rolling(20, min_periods=20).mean().values
    _kc_upper = _kc_mid + 1.5 * _kc_atr; _kc_lower = _kc_mid - 1.5 * _kc_atr
    _bb_mid = close_s.rolling(20, min_periods=20).mean().values
    _bb_std = close_s.rolling(20, min_periods=20).std().values
    _bb_upper = _bb_mid + 2 * _bb_std; _bb_lower = _bb_mid - 2 * _bb_std
    regime_ok = np.array([not np.isnan(_kc_upper[i]) and _bb_upper[i] < _kc_upper[i] for i in range(n)])

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
