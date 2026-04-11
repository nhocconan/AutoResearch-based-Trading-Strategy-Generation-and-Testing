#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w OHLC for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (based on previous week's range)
    range_1w = high_1w - low_1w
    camarilla_r3 = close_1w + (range_1w * 1.1 / 4)
    camarilla_s3 = close_1w - (range_1w * 1.1 / 4)
    camarilla_pivot = close_1w  # Previous week's close
    
    # Calculate 1w average volume (10-period) for volume filter
    volume_1w = df_1w['volume'].values
    vol_avg_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align all 1w indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    vol_avg_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 10 to ensure sufficient data for volume average
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_avg_10_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1w volume (aligned)
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        vol_spike = vol_1w_current > 2.0 * vol_avg_10_1w_aligned[i]  # 100% above average
        
        price = close[i]
        
        # Long when price breaks above Camarilla R3 with volume spike
        long_breakout = price > camarilla_r3_aligned[i]
        long_signal = long_breakout and vol_spike
        
        # Short when price breaks below Camarilla S3 with volume spike
        short_breakout = price < camarilla_s3_aligned[i]
        short_signal = short_breakout and vol_spike
        
        # Exit when price returns to the Camarilla pivot (previous week's close)
        exit_long = price < camarilla_pivot_aligned[i]
        exit_short = price > camarilla_pivot_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals