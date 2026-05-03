#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla R3/S3 fade + volume confirmation
# Long when price breaks above Donchian(20) high AND price < 1d Camarilla R3 (fade from resistance)
# Short when price breaks below Donchian(20) low AND price > 1d Camarilla S3 (fade from support)
# Volume confirmation required (2.0x 20-period average) to avoid false breakouts
# Uses 1d Camarilla levels for mean reversion edge within 6h trend structure
# Designed for low trade frequency (12-37/year on 6h) to minimize fee drag

name = "6h_Donchian20_CamarillaR3S3_Fade_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 4.0
    s3_1d = pivot_1d - range_1d * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Donchian(20) channels on 6h
    # Upper band: highest high over last 20 periods
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: lowest low over last 20 periods
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(20 for Donchian, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper band AND price < R3 (fade from resistance) + volume spike
            if (close[i] > upper_band[i] and close[i] < r3_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band AND price > S3 (fade from support) + volume spike
            elif (close[i] < lower_band[i] and close[i] > s3_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower band OR price > R3 (stop loss at resistance)
            if (close[i] < lower_band[i] or close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper band OR price < S3 (stop loss at support)
            if (close[i] > upper_band[i] or close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals