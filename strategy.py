#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from daily data with volume confirmation
# Fade at R3/S3 levels, breakout continuation at R4/S4 levels
# Works in both bull and bear markets by using price action around key intraday levels
# Camarilla levels provide natural support/resistance; volume confirms institutional interest
# Low frequency design targets 12-37 trades per year to minimize fee drag

name = "6h_camarilla_pivot_1d_volume_v1"
timeframe = "6h"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    r4 = close_1d + 1.5 * range_1d
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d
    s4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (24-period average on 6h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Price levels
        r4_level = r4_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        s4_level = s4_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price breaks below S3 (mean reversion failure) or reaches R4 (take profit)
            if close[i] < s3_level or close[i] > r4_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price breaks above R3 (mean reversion failure) or reaches S4 (take profit)
            if close[i] > r3_level or close[i] < s4_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Fade at R3/S3: sell at R3, buy at S3 with volume confirmation
            # Breakout continuation: buy above R4, sell below S4 with volume confirmation
            
            # Long: buy at S3 bounce or break above R4
            if (abs(close[i] - s3_level) < 0.001 * s3_level and vol_confirm) or \
               (close[i] > r4_level and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: sell at R3 bounce or break below S4
            elif (abs(close[i] - r3_level) < 0.001 * r3_level and vol_confirm) or \
                 (close[i] < s4_level and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals