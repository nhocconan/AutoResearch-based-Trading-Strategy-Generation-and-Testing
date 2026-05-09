#!/usr/bin/env python3
name = "1D_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA for weekly trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly EMA data not ready
        if np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels based on previous day's range
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla R3 and S3 levels
        range_val = prev_high - prev_low
        r3 = prev_close + (range_val * 1.1 / 2)
        s3 = prev_close - (range_val * 1.1 / 2)
        
        # Determine trend from weekly EMA
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average volume
        avg_volume = np.mean(volume[i-20:i])
        volume_confirm = volume[i] > avg_volume * 2.0
        
        if position == 0:
            # Enter long: price breaks above R3 + uptrend + volume confirmation
            if close[i] > r3 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + downtrend + volume confirmation
            elif close[i] < s3 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below previous day's close (mean reversion)
            if close[i] < prev_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above previous day's close
            if close[i] > prev_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals