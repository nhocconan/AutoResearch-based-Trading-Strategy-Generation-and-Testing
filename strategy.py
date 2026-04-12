#!/usr/bin/env python3
"""
12h_1w_1d_camarilla_breakout_volume_regime
Hypothesis: Weekly trend filter + daily Camarilla levels with volume confirmation on 12h timeframe.
Works in bull/bear by using weekly trend direction to filter long/short signals, reducing false breakouts.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

name = "12h_1w_1d_camarilla_breakout_volume_regime"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema_50_1w
    
    # Get daily data for Camarilla levels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's range (for Camarilla)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Daily volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x daily average volume
        vol_confirm = volume[i] > (vol_ma_aligned[i] * 1.5)
        
        # Long entry: weekly uptrend + price breaks above R4 + volume
        if (weekly_uptrend_aligned[i] and close[i] > r4_aligned[i] and 
            vol_confirm and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: weekly downtrend + price breaks below S4 + volume
        elif (not weekly_uptrend_aligned[i] and close[i] < s4_aligned[i] and 
              vol_confirm and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals