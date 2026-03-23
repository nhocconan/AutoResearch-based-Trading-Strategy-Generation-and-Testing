#!/usr/bin/env python3
"""Auto-generated: supertrend trend + cci entry + aroon_filter regime on 12h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_supertrend_cci_aroon_filter_12h_v1"
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

    tr = np.zeros(n)
    for i in range(1, n): tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr_st = pd.Series(tr).rolling(10, min_periods=10).mean().values
    hl2 = (high + low) / 2; upper = hl2 + 2.0*atr_st; lower = hl2 - 2.0*atr_st
    trend = np.zeros(n); fu = np.full(n, np.nan); fl = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(upper[i]): trend[i]=trend[i-1]; continue
        fl[i] = max(lower[i], fl[i-1]) if not np.isnan(fl[i-1]) and close[i-1]>fl[i-1] else lower[i]
        fu[i] = min(upper[i], fu[i-1]) if not np.isnan(fu[i-1]) and close[i-1]<fu[i-1] else upper[i]
        if close[i]>fu[i]: trend[i]=1
        elif close[i]<fl[i]: trend[i]=-1
        else: trend[i]=trend[i-1]

    # ENTRY filter

    tp = (high + low + close) / 3
    tp_sma = pd.Series(tp).rolling(20, min_periods=20).mean().values
    tp_mad = pd.Series(tp).rolling(20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean()))).values
    cci = np.where(tp_mad > 0, (tp - tp_sma) / (0.015 * tp_mad), 0)
    entry_ok_long = cci < -100
    entry_ok_short = cci > 50

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
