#!/usr/bin/env python3
"""Auto-generated: parabolic_sar trend + session_filter entry + bbw_regime regime on 12h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_parabolic_sar_session_filter_bbw_regime_12h_v1"
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

    sar = np.zeros(n); sar[0] = low[0]; af = 0.02; ep = high[0]; is_long = True
    trend = np.zeros(n)
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if is_long:
            if low[i] < sar[i]: is_long = False; sar[i] = ep; ep = low[i]; af = 0.02
            else:
                if high[i] > ep: ep = high[i]; af = min(af + 0.02, 0.20)
        else:
            if high[i] > sar[i]: is_long = True; sar[i] = ep; ep = high[i]; af = 0.02
            else:
                if low[i] < ep: ep = low[i]; af = min(af + 0.02, 0.20)
        trend[i] = 1.0 if is_long else -1.0

    # ENTRY filter

    # Time-of-day seasonality: best crypto returns 20:00-00:00 UTC
    _hours = np.zeros(n)
    if 'open_time' in prices.columns:
        _hours = pd.to_datetime(prices['open_time']).dt.hour.values
    entry_ok_long = np.array([18 <= int(_hours[i]) <= 23 or int(_hours[i]) < 2 for i in range(n)])
    entry_ok_short = entry_ok_long.copy()

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
