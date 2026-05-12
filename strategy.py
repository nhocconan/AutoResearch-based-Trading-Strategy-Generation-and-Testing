#!/usr/bin/env python3
# 6h_1w_1d_WeeklyPivot_Trend_Breakout
# Hypothesis: Combines weekly pivot levels (R2/S2) for breakout entries on 6h timeframe with 1d EMA100 trend filter.
# Weekly pivots provide strong support/resistance levels that hold across market regimes.
# Trend filter ensures trades align with daily direction, reducing whipsaws in sideways markets.
# Volume confirmation (>1.8x 30-period average) ensures institutional participation.
# Designed for low trade frequency (<300 total 6h trades) to minimize fee drag.
# Works in bull/bear markets by following daily trend while using weekly pivots for precise entries.

name = "6h_1w_1d_WeeklyPivot_Trend_Breakout"
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
    
    # Volume spike: >1.8x 30-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Standard pivot point formulas
    pivot = (prev_high + prev_low + prev_close) / 3.0
    weekly_r2 = pivot + 2 * (prev_high - prev_low)
    weekly_s2 = pivot - 2 * (prev_high - prev_low)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align weekly and daily indicators to 6h timeframe
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or
            np.isnan(ema_100_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Weekly R2 + volume spike + price above daily EMA100
            if (close[i] > weekly_r2_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_100_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Weekly S2 + volume spike + price below daily EMA100
            elif (close[i] < weekly_s2_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_100_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Weekly R2 OR closes below daily EMA100
            if (close[i] < weekly_r2_aligned[i]) or \
               close[i] < ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Weekly S2 OR closes above daily EMA100
            if (close[i] > weekly_s2_aligned[i]) or \
               close[i] > ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals