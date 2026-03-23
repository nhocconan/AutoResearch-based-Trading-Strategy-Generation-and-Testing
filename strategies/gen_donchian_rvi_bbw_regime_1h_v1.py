#!/usr/bin/env python3
"""Auto-generated: donchian trend + rvi entry + bbw_regime regime on 1h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_donchian_rvi_bbw_regime_1h_v1"
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

    don_h = pd.Series(high).rolling(15, min_periods=15).max().values
    don_l = pd.Series(low).rolling(15, min_periods=15).min().values
    trend = np.zeros(n)
    for i in range(15+1, n):
        if close[i] > don_h[i-1]: trend[i] = 1.0
        elif close[i] < don_l[i-1]: trend[i] = -1.0
        else: trend[i] = trend[i-1]

    # ENTRY filter

    _rvi_num = pd.Series((close - prices["open"].values) if "open" in prices.columns else np.zeros(n)).rolling(10,min_periods=10).mean().values
    _rvi_den = pd.Series(high - low).rolling(10,min_periods=10).mean().values
    _rvi = np.where(_rvi_den>0, _rvi_num/_rvi_den, 0)
    _rvi_sig = pd.Series(_rvi).rolling(4,min_periods=4).mean().values
    entry_ok_long = np.array([_rvi[i]>_rvi_sig[i] for i in range(n)])
    entry_ok_short = np.array([_rvi[i]<_rvi_sig[i] for i in range(n)])

    # REGIME filter

    _sma = close_s.rolling(20, min_periods=20).mean().values
    _std = close_s.rolling(20, min_periods=20).std().values
    _bbw = np.where(_sma > 0, _std / _sma, 0)
    _bbw_pct = pd.Series(_bbw).rolling(100, min_periods=50).rank(pct=True).values
    regime_ok = np.array([not np.isnan(_bbw_pct[i]) and _bbw_pct[i] < 0.7 and _bbw_pct[i] > 0.1 for i in range(n)])

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
