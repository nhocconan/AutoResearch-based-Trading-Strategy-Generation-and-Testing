#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R3S3_Fade_Volume_Stretch"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels: R3, S3, R4, S4
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels: R3, S3, R4, S4
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align daily Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: current volume > 1.3x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Price stretch from S3/R3: how far price has moved beyond S3/R3 toward S4/R4
    # For longs: when price is between S3 and S4, stretch = (price - S3) / (S4 - S3)
    # For shorts: when price is between R3 and R4, stretch = (R4 - price) / (R4 - R3)
    stretch_long = np.zeros(n)
    stretch_short = np.zeros(n)
    
    for i in range(n):
        if not np.isnan(s3_aligned[i]) and not np.isnan(s4_aligned[i]) and s4_aligned[i] != s3_aligned[i]:
            if close[i] >= s3_aligned[i] and close[i] <= s4_aligned[i]:
                stretch_long[i] = (close[i] - s3_aligned[i]) / (s4_aligned[i] - s3_aligned[i])
        if not np.isnan(r3_aligned[i]) and not np.isnan(r4_aligned[i]) and r4_aligned[i] != r3_aligned[i]:
            if close[i] >= r3_aligned[i] and close[i] <= r4_aligned[i]:
                stretch_short[i] = (r4_aligned[i] - close[i]) / (r4_aligned[i] - r3_aligned[i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long fade at S3: price rejected from S3 downward with volume
            if price <= s3_aligned[i] and stretch_long[i] < 0.3 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short fade at R3: price rejected from R3 upward with volume
            elif price >= r3_aligned[i] and stretch_short[i] < 0.3 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price moves back to S3 (mean reversion) or reaches S4 (breakout)
            if price >= s3_aligned[i] or price <= s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price moves back to R3 (mean reversion) or reaches R4 (breakout)
            if price <= r3_aligned[i] or price >= r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals