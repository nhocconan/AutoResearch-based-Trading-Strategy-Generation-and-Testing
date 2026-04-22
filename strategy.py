#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's HLC (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Pivot points
    pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pp_1d - prev_low_1d
    s1_1d = 2 * pp_1d - prev_high_1d
    r2_1d = pp_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pp_1d - (prev_high_1d - prev_low_1d)
    r3_1d = prev_high_1d + 2 * (pp_1d - prev_low_1d)
    s3_1d = prev_low_1d - 2 * (prev_high_1d - pp_1d)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period average on 6h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        ema50 = ema50_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume + above EMA50
            if price > r3 and vol > 2.0 * vol_ma and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume + below EMA50
            elif price < s3 and vol > 2.0 * vol_ma and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through S1/R1
            if position == 1 and price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Pivot_R3_S3_Breakout_1dEMA50_Volume_Spike"
timeframe = "6h"
leverage = 1.0