#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: Range = High - Low
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.0833)
    r2 = close_1d + (range_1d * 1.1666)
    r3 = close_1d + (range_1d * 1.2500)
    r4 = close_1d + (range_1d * 1.3333)
    s1 = close_1d - (range_1d * 1.0833)
    s2 = close_1d - (range_1d * 1.1666)
    s3 = close_1d - (range_1d * 1.2500)
    s4 = close_1d - (range_1d * 1.3333)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r1_shifted = np.roll(r1, 1)
    r2_shifted = np.roll(r2, 1)
    r3_shifted = np.roll(r3, 1)
    r4_shifted = np.roll(r4, 1)
    s1_shifted = np.roll(s1, 1)
    s2_shifted = np.roll(s2, 1)
    s3_shifted = np.roll(s3, 1)
    s4_shifted = np.roll(s4, 1)
    r1_shifted[0] = np.nan
    r2_shifted[0] = np.nan
    r3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    s2_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_shifted)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_shifted)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_shifted)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_shifted)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_shifted)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_shifted)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: volume > 1.8x 20-period EMA (stricter to reduce trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or
            np.isnan(s1_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and above daily EMA34
            if (price > r3_6h[i] and vol_spike[i] and price > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and below daily EMA34
            elif (price < s3_6h[i] and vol_spike[i] and price < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S3 (mean reversion to support)
            if price < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R3 (mean reversion to resistance)
            if price > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals