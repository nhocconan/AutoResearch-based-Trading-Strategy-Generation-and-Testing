#!/usr/bin/env python3
"""
1d_1W_WeeklyPivot_HighLow_Breakout
Hypothesis: Daily breakouts above weekly pivot high/low with volume confirmation.
Long when price breaks above weekly pivot high + volume spike in uptrend (price > weekly EMA20).
Short when price breaks below weekly pivot low + volume spike in downtrend (price < weekly EMA20).
Weekly pivot provides significant support/resistance; breakouts with volume indicate strong moves.
Works in bull via trend continuation breaks and in bear via sharp reversals at key weekly levels.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
"""

name = "1d_1W_WeeklyPivot_HighLow_Breakout"
timeframe = "1d"
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
    
    # Volume spike: >1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly pivot high and low from previous week
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    range_1w = prev_high_1w - prev_low_1w
    pivot_high_1w = prev_high_1w + 0.5 * range_1w  # Weekly resistance
    pivot_low_1w = prev_low_1w - 0.5 * range_1w   # Weekly support
    
    # Align weekly levels to daily timeframe
    pivot_high_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_high_1w)
    pivot_low_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_low_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(pivot_high_1w_aligned[i]) or 
            np.isnan(pivot_low_1w_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly pivot high + volume spike + price above weekly EMA20 (uptrend)
            if (close[i] > pivot_high_1w_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly pivot low + volume spike + price below weekly EMA20 (downtrend)
            elif (close[i] < pivot_low_1w_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below weekly pivot high OR below weekly EMA20
            if close[i] < pivot_high_1w_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above weekly pivot low OR above weekly EMA20
            if close[i] > pivot_low_1w_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals