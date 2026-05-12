#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1dTrend_Volume
Hypothesis: Trade Camarilla pivot breakouts (H5/L5) on 12h timeframe with 1d trend confirmation.
Camarilla levels provide precise entry points in ranging markets, while 1d trend filter ensures
alignment with higher timeframe direction. Volume spike confirms institutional participation.
Designed for low trade frequency (target 20-50 trades/year) to minimize fee drag.
Works in both bull and bear markets by following 1d trend direction.
"""

name = "12h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    h5 = pivot + (range_val * 1.1 / 2)
    h4 = pivot + (range_val * 1.1)
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    l4 = pivot - (range_val * 1.1)
    l5 = pivot - (range_val * 1.1 / 2)
    return h5, h4, h3, l3, l4, l5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivot and EMA trend ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily Camarilla levels from previous day
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Previous day's Camarilla levels (avoid look-ahead)
    h5, h4, h3, l3, l4, l5 = calculate_camarilla(d_high, d_low, d_close)
    h5_prev = np.roll(h5, 1)
    h4_prev = np.roll(h4, 1)
    h3_prev = np.roll(h3, 1)
    l3_prev = np.roll(l3, 1)
    l4_prev = np.roll(l4, 1)
    l5_prev = np.roll(l5, 1)
    # Set first values to avoid look-ahead on first bar
    h5_prev[0] = h5[0]
    h4_prev[0] = h4[0]
    h3_prev[0] = h3[0]
    l3_prev[0] = l3[0]
    l4_prev[0] = l4[0]
    l5_prev[0] = l5[0]
    
    # Align Camarilla levels to 12h timeframe
    h5_12h = align_htf_to_ltf(prices, df_1d, h5_prev)
    h4_12h = align_htf_to_ltf(prices, df_1d, h4_prev)
    h3_12h = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3_prev)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4_prev)
    l5_12h = align_htf_to_ltf(prices, df_1d, l5_prev)

    # Calculate daily EMA34 for trend filter
    d_ema34 = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    d_ema34_12h = align_htf_to_ltf(prices, df_1d, d_ema34)

    # 12h volume spike: current > 2.0x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(h5_12h[i]) or np.isnan(l5_12h[i]) or 
            np.isnan(d_ema34_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above H5 with volume spike and above daily EMA34 (uptrend)
            if (high[i] > h5_12h[i] and 
                volume_spike[i] and 
                close[i] > d_ema34_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below L5 with volume spike and below daily EMA34 (downtrend)
            elif (low[i] < l5_12h[i] and 
                  volume_spike[i] and 
                  close[i] < d_ema34_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below L3 or trend changes
            if (low[i] < l3_12h[i]) or (close[i] < d_ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above H3 or trend changes
            if (high[i] > h3_12h[i]) or (close[i] > d_ema34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals