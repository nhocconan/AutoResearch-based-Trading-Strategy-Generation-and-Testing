#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla formulas (based on previous day's range)
    # R4 = close + (high - low) * 1.5
    # R3 = close + (high - low) * 1.25
    # R2 = close + (high - low) * 1.1666
    # R1 = close + (high - low) * 1.0833
    # S1 = close - (high - low) * 1.0833
    # S2 = close - (high - low) * 1.1666
    # S3 = close - (high - low) * 1.25
    # S4 = close - (high - low) * 1.5
    
    range_1d = high_1d - low_1d
    r3 = close_1d_vals + range_1d * 1.25
    s3 = close_1d_vals - range_1d * 1.25
    r4 = close_1d_vals + range_1d * 1.5
    s4 = close_1d_vals - range_1d * 1.5
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection on 6h (20-period MA)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume, in uptrend
            if (close[i] > r3_aligned[i] and vol_ok and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume, in downtrend
            elif (close[i] < s3_aligned[i] and vol_ok and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below R3
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above S3
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals