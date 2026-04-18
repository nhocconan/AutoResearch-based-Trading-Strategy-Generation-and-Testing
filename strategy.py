#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 12h timeframe. 
# Long when price breaks above R4 with volume confirmation.
# Short when price breaks below S4 with volume confirmation.
# Exit when price returns to R3/S3 levels.
# Camarilla levels from higher timeframe (12h) provide institutional reference points.
# Volume surge confirms institutional participation.
# Designed for ~15-25 trades/year per symbol.
name = "6h_12hCamarilla_R4_S4_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formula: range = high - low
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    range_12h = high_12h - low_12h
    r4_12h = close_12h + (range_12h * 1.1 / 2)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    s4_12h = close_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (wait for 12h bar close)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 6h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r4_level = r4_12h_aligned[i]
        r3_level = r3_12h_aligned[i]
        s3_level = s3_12h_aligned[i]
        s4_level = s4_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume
            if close_val > r4_level and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume
            elif close_val < s4_level and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to R3 level
            if close_val <= r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to S3 level
            if close_val >= s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals