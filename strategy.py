#!/usr/bin/env python3
"""
12h_VWAP_Cross_1dTrend_Volume
Hypothesis: Price crossing VWAP on 12h with 1-day trend filter and volume confirmation.
Uses VWAP as a dynamic support/resistance level that adapts to market conditions.
Designed for 15-30 trades/year on 12h timeframe to work in both bull and bear markets
by capturing mean-reversion in range markets and trend continuation in trending markets.
"""

name = "12h_VWAP_Cross_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate VWAP for 12h period (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)

    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 12h volume > 1.3x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        vwap_val = vwap[i]
        ema34_val = ema34_1d_aligned[i]
        vol_avg_val = vol_avg_30[i]

        if np.isnan(vwap_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above VWAP + uptrend + volume confirmation
            if close[i] > vwap_val and close[i] > ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below VWAP + downtrend + volume confirmation
            elif close[i] < vwap_val and close[i] < ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below VWAP or trend changes
            if close[i] < vwap_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above VWAP or trend changes
            if close[i] > vwap_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals