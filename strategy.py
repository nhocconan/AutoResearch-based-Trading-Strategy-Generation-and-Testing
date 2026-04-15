#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# Uses weekly trend direction to filter breakouts of the prior day's high/low.
# Long when price breaks above prior day's high AND weekly trend is up.
# Short when price breaks below prior day's low AND weekly trend is down.
# Volume confirmation ensures breakout strength. Designed for low trade frequency (<25/year).
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for prior day's high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Prior day's high/low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Align prior day's levels to 1d timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Weekly trend: 20-period EMA slope (positive = up, negative = down)
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slope = np.diff(ema_20, prepend=ema_20[0])
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or 
            np.isnan(prev_low_1d_aligned[i]) or 
            np.isnan(ema_slope_aligned[i])):
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day median
        vol_median = np.median(volume[max(0, i-20):i+1])
        vol_ok = volume[i] > 1.5 * vol_median
        
        # Long: break above prior day's high + up trend + volume
        if (close[i] > prev_high_1d_aligned[i] and 
            ema_slope_aligned[i] > 0 and 
            vol_ok and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: break below prior day's low + down trend + volume
        elif (close[i] < prev_low_1d_aligned[i] and 
              ema_slope_aligned[i] < 0 and 
              vol_ok and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or trend change
        elif position == 1 and (close[i] < prev_low_1d_aligned[i] or ema_slope_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_high_1d_aligned[i] or ema_slope_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0