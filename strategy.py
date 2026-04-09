#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels with volume confirmation
# 4h trend direction from Camarilla S3/R3, 1h entry timing on S4/R4 breakouts with volume filter
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Fixed position size 0.20 to control risk and minimize fee churn
# Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
# Uses higher timeframe structure for direction, lower timeframe for precise entries

name = "1h_4h_1d_camarilla_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (trend direction)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    camarilla_s3_4h = close_4h - range_4h * 1.1 / 4.0  # S3 = support level
    camarilla_r3_4h = close_4h + range_4h * 1.1 / 4.0  # R3 = resistance level
    
    # Align 4h Camarilla levels to 1h timeframe
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (entry timing)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_s4_1d = close_1d - range_1d * 1.1 / 2.0  # S4 = strong support
    camarilla_r4_1d = close_1d + range_1d * 1.1 / 2.0  # R4 = strong resistance
    
    # Align 1d Camarilla levels to 1h timeframe
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(s3_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation: current 1h volume > 1.5x average 1h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if price breaks below 4h S3 (trend reversal)
            if close[i] < s3_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit short if price breaks above 4h R3 (trend reversal)
            if close[i] > r3_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for 1d Camarilla S4/R4 breakouts with volume confirmation and session filter
            if in_session and volume_confirmed:
                if close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals