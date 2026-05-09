#!/usr/bin/env python3
name = "1h_Camarilla_Pivot_4dTrend_VolumeSpike"
timeframe = "1h"
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
    
    # Get 4d data for trend filter and Camarilla pivot levels
    df_4d = get_htf_data(prices, '4d')
    if len(df_4d) < 30:
        return np.zeros(n)
    
    close_4d = df_4d['close'].values
    high_4d = df_4d['high'].values
    low_4d = df_4d['low'].values
    
    # Calculate 4d EMA100 for trend filter
    if len(close_4d) >= 100:
        ema100_4d = pd.Series(close_4d).ewm(span=100, adjust=False, min_periods=100).mean().values
    else:
        ema100_4d = np.full_like(close_4d, np.nan)
    
    # Align 4d EMA100 to 1h timeframe
    ema100_4d_aligned = align_htf_to_ltf(prices, df_4d, ema100_4d)
    
    # Calculate Camarilla pivot levels from previous 4d
    camarilla_h3 = np.full_like(close_4d, np.nan)
    camarilla_l3 = np.full_like(close_4d, np.nan)
    
    for i in range(len(close_4d)):
        if i >= 1:  # Need previous 4d's data
            prev_high = high_4d[i-1]
            prev_low = low_4d[i-1]
            prev_close = close_4d[i-1]
            range_ = prev_high - prev_low
            camarilla_h3[i] = prev_close + 1.1 * range_ / 4
            camarilla_l3[i] = prev_close - 1.1 * range_ / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4d, camarilla_l3)
    
    # Get 1h data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h volume EMA20
    if len(volume_1h) >= 20:
        vol_ema20_1h = pd.Series(volume_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        vol_ema20_1h = np.full_like(volume_1h, np.nan)
    
    # Align 1h volume EMA20 to 1h timeframe
    vol_ema20_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ema20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(1, 100, 20)  # Need Camarilla (1 day), EMA100, volume EMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema100_4d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ema20_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 4d EMA100
        uptrend = close[i] > ema100_4d_aligned[i]
        # Downtrend: price below 4d EMA100
        downtrend = close[i] < ema100_4d_aligned[i]
        # Volume surge: current volume > 1.5x 1h volume EMA20
        volume_surge = volume[i] > vol_ema20_1h_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price touches/breaks above Camarilla H3 + volume surge
            if uptrend and close[i] >= camarilla_h3_aligned[i] and volume_surge:
                signals[i] = 0.20
                position = 1
            # Enter short: Downtrend + price touches/breaks below Camarilla L3 + volume surge
            elif downtrend and close[i] <= camarilla_l3_aligned[i] and volume_surge:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price touches/breaks below Camarilla L3
            if not uptrend or close[i] <= camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Trend turns up OR price touches/breaks above Camarilla H3
            if not downtrend or close[i] >= camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals