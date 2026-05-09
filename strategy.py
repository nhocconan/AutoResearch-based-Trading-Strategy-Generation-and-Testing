#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_VolumeSpike_Camarilla_Pivot_Breakout"
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
    
    # Daily Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = High - Low
    range_1d = high_1d - low_1d
    # Resistance levels
    r1 = close_1d + (range_1d * 1.0833)
    r2 = close_1d + (range_1d * 1.1666)
    r3 = close_1d + (range_1d * 1.2500)
    r4 = close_1d + (range_1d * 1.3333)
    # Support levels
    s1 = close_1d - (range_1d * 1.0833)
    s2 = close_1d - (range_1d * 1.1666)
    s3 = close_1d - (range_1d * 1.2500)
    s4 = close_1d - (range_1d * 1.3333)
    
    # Align Camarilla levels to 6h timeframe (need previous day's levels)
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    r1_shifted = np.roll(r1, 1)
    r2_shifted = np.roll(r2, 1)
    r3_shifted = np.roll(r3, 1)
    r4_shifted = np.roll(r4, 1)
    s1_shifted = np.roll(s1, 1)
    s2_shifted = np.roll(s2, 1)
    s3_shifted = np.roll(s3, 1)
    s4_shifted = np.roll(s4, 1)
    # Set first day's values to NaN (no previous day)
    r1_shifted[0] = np.nan
    r2_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    s2_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_shifted)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_shifted)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_shifted)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_shifted)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_shifted)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_shifted)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # Volume spike filter: volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or
            np.isnan(s1_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume spike
            if (price > r4_6h[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with volume spike
            elif (price < s4_6h[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R3 (mean reversion)
            if price < r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S3 (mean reversion)
            if price > s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals