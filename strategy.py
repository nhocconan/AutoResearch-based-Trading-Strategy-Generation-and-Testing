#!/usr/bin/env python3
name = "1h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (using previous day's OHLC)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = close_1d + (range_1d * 1.1 / 4)
    s3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align daily levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: current volume > 1.5x 24-period average volume
    volume = prices['volume'].values
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > volume_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check if we're in trading session (08-20 UTC)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 + uptrend + volume confirmation
            if (prices['close'].iloc[i] > r3_aligned[i] and 
                prices['close'].iloc[i] > ema34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S3 + downtrend + volume confirmation
            elif (prices['close'].iloc[i] < s3_aligned[i] and 
                  prices['close'].iloc[i] < ema34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend reverses
            if (prices['close'].iloc[i] < s3_aligned[i] or 
                prices['close'].iloc[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend reverses
            if (prices['close'].iloc[i] > r3_aligned[i] or 
                prices['close'].iloc[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals