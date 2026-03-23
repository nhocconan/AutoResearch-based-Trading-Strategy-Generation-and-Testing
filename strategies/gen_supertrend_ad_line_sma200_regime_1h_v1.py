#!/usr/bin/env python3
"""Auto-generated: supertrend trend + ad_line entry + sma200_regime regime on 1h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_supertrend_ad_line_sma200_regime_1h_v1"
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

    _clv = np.where(high-low>0, (2*close-low-high)/(high-low), 0)
    _ad = np.cumsum(_clv * volume)
    _ad_ema = pd.Series(_ad).ewm(span=21,min_periods=21,adjust=False).mean().values
    entry_ok_long = np.array([_ad[i]>_ad_ema[i] if not np.isnan(_ad_ema[i]) else False for i in range(n)])
    entry_ok_short = np.array([_ad[i]<_ad_ema[i] if not np.isnan(_ad_ema[i]) else False for i in range(n)])

    # REGIME filter

    _sma200 = close_s.rolling(200, min_periods=200).mean().values
    regime_ok = np.array([not np.isnan(_sma200[i]) and close[i] > _sma200[i] for i in range(n)])

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
