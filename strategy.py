#!/usr/bin/env python3
"""
6h_Liquidity_Imbalance_Reversion
Hypothesis: Exploit mean reversion from intraday liquidity imbalances using 6-hour price action filtered by 12-hour trend and volume exhaustion. Works in both bull and bear markets by fading extreme deviations from the 12-hour VWAP when volume dries up, capturing reversals after stop hunts or false breakouts. Target: 25-40 trades/year.
"""

name = "6h_Liquidity_Imbalance_Reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for trend and VWAP (call once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values

    # 12h VWAP (volume-weighted average price)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vwap_num = np.cumsum(typical_price_12h * volume_12h)
    vwap_den = np.cumsum(volume_12h)
    vwap_12h = vwap_num / vwap_den
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)

    # 12h trend: EMA20 of close
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)

    # 6h volatility: ATR(14) for dynamic thresholds
    tr = np.maximum(np.abs(high[1:] - low[:-1]),
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    # 6h volume exhaustion: volume < 0.6x 20-period average (low liquidity)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        vwap = vwap_12h_aligned[i]
        ema20 = ema20_12h_aligned[i]
        vol_exhaust = vol_avg_20[i]
        atr = atr14[i]

        if np.isnan(vwap) or np.isnan(ema20) or np.isnan(vol_exhaust) or np.isnan(atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Dynamic deviation threshold: 1.2 * ATR
        dev_threshold = 1.2 * atr
        deviation = close[i] - vwap

        if position == 0:
            # LONG: price significantly below VWAP + volume exhaustion + price above 12h EMA (bullish bias)
            if deviation < -dev_threshold and volume[i] < vol_exhaust * 0.6 and close[i] > ema20:
                signals[i] = 0.25
                position = 1
            # SHORT: price significantly above VWAP + volume exhaustion + price below 12h EMA (bearish bias)
            elif deviation > dev_threshold and volume[i] < vol_exhaust * 0.6 and close[i] < ema20:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to VWAP or volume returns (liquidity returns)
            if deviation > -0.3 * dev_threshold or volume[i] > vol_exhaust * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to VWAP or volume returns
            if deviation < 0.3 * dev_threshold or volume[i] > vol_exhaust * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals