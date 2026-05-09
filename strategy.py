#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
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
    
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_shifted)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_shifted)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_shifted)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_shifted)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_shifted)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_shifted)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume spike filter: volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or
            np.isnan(s1_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume spike and 12h uptrend
            if (price > r3_4h[i] and vol_spike[i] and price > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with volume spike and 12h downtrend
            elif (price < s3_4h[i] and vol_spike[i] and price < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R2 (mean reversion)
            if price < r2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S2 (mean reversion)
            if price > s2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals