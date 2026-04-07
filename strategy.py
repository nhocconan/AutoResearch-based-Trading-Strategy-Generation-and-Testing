#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Weekly Camarilla pivot levels act as strong support/resistance in trending markets. 
When price approaches weekly R4/S4 levels with volume confirmation and daily trend alignment, 
it often continues in the direction of the breakout. Works in both bull and bear markets by 
following the trend defined by the weekly EMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week)
    weekly_close = df_1w['close'].shift(1).values
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    weekly_range = weekly_high - weekly_low
    
    # Camarilla levels
    r4 = weekly_close + weekly_range * 1.1 / 2
    r3 = weekly_close + weekly_range * 1.1 / 4
    r2 = weekly_close + weekly_range * 1.1 / 6
    r1 = weekly_close + weekly_range * 1.1 / 12
    s1 = weekly_close - weekly_range * 1.1 / 12
    s2 = weekly_close - weekly_range * 1.1 / 6
    s3 = weekly_close - weekly_range * 1.1 / 4
    s4 = weekly_close - weekly_range * 1.1 / 2
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean().values
    
    # Align all weekly data to daily timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 or trend turns bearish
            if close[i] <= r3_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above S3 or trend turns bullish
            if close[i] >= s3_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or breaks above R4 with volume and bullish trend
            if (close[i] >= r4_aligned[i] and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or breaks below S4 with volume and bearish trend
            elif (close[i] <= s4_aligned[i] and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals