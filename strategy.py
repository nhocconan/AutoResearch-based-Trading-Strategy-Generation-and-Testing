#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels (S3/S4 for long, R3/R4 for short) with 12h trend filter and volume confirmation
# Uses institutional pivot levels from daily data for high-probability reversals
# Requires 12h EMA trend alignment and volume spike to reduce false signals
# Target: 20-50 total trades over 4 years = 5-12/year

name = "4h_Camarilla_S3R3_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with original arrays
    
    # Camarilla multiplier
    # R4 = C + ((H-L) * 1.500), R3 = C + ((H-L) * 1.250)
    # S3 = C - ((H-L) * 1.250), S4 = C - ((H-L) * 1.500)
    hl_range_1d = high_1d - low_1d
    r4_1d = close_1d + (hl_range_1d * 1.500)
    r3_1d = close_1d + (hl_range_1d * 1.250)
    s3_1d = close_1d - (hl_range_1d * 1.250)
    s4_1d = close_1d - (hl_range_1d * 1.500)
    
    # Align Camarilla levels to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        ema12h_val = ema50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price at S3 level + uptrend + volume spike
            if (close[i] <= s3_val * 1.002 and  # allow small buffer for touches
                close[i] > ema12h_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price at R3 level + downtrend + volume spike
            elif (close[i] >= r3_val * 0.998 and  # allow small buffer for touches
                  close[i] < ema12h_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches S4 (target) or trend turns bearish
            if (close[i] <= s4_1d_aligned[i] * 1.002 or close[i] < ema12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches R4 (target) or trend turns bullish
            if (close[i] >= r4_1d_aligned[i] * 0.998 or close[i] > ema12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals