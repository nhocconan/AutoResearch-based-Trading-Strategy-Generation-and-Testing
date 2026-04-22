#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for weekly pivot points and weekly trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Previous week's high, low, close for weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot = (high_1w + low_1w + close_1w) / 3
    range_ = high_1w - low_1w
    r1 = pivot + range_  # Resistance level 1
    s1 = pivot - range_  # Support level 1
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50 = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume spike AND above 1w EMA50 (uptrend)
            if (close[i] > r1_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume spike AND below 1w EMA50 (downtrend)
            elif (close[i] < s1_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite weekly pivot level
            if position == 1:
                # Exit long: Price closes below weekly S1
                if not np.isnan(s1_aligned[i]) and close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above weekly R1
                if not np.isnan(r1_aligned[i]) and close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyPivot_R1_S1_Breakout_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0