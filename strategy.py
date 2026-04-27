#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels and daily EMA trend filter with volume confirmation.
# Weekly pivot levels (R2/S2) act as major support/resistance; breakouts indicate strong momentum.
# Uses daily EMA34 for trend filter to avoid counter-trend trades.
# Volume spike (>1.5x 30-period average) confirms institutional participation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (breakout continuations) and bear markets (breakdown continuations).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels (R2/S2)
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R2 = Pivot + (Range * 1.1)
    # S2 = Pivot - (Range * 1.1)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r2_1w = pivot_1w + (range_1w * 1.1)
    s2_1w = pivot_1w - (range_1w * 1.1)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R2 with uptrend and volume
        if (close[i] > r2_1w_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short breakdown: price breaks below S2 with downtrend and volume
        elif (close[i] < s2_1w_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions
        elif position == 1 and close[i] <= ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R2S2_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0