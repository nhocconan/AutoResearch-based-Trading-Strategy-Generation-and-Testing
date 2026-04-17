#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.3x average AND 12h EMA34 > EMA89 (uptrend).
Short when price breaks below Camarilla S1 AND volume > 1.3x average AND 12h EMA34 < EMA89 (downtrend).
Exit when price reverts to Camarilla central pivot point (CPP) OR 12h EMA flips direction.
Uses 4h for Camarilla calculation and 12h for EMA filter to reduce whipsaw and capture medium-term trends.
Volume confirmation filters fakeouts, EMA filter avoids ranging markets.
Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets (captures uptrends) and bear markets (captures downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels on 4h timeframe (using previous 4h bar's OHLC)
    # Camarilla uses previous period's range
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    
    # First bar: use current values (will be filtered by min_periods later)
    prev_close_4h[0] = close_4h[0]
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    
    # Calculate range
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla levels
    # R4 = close + (high-low)*1.5/2, R3 = close + (high-low)*1.25/2, R2 = close + (high-low)*1.1/2, R1 = close + (high-low)*1.05/2
    # S1 = close - (high-low)*1.05/2, S2 = close - (high-low)*1.1/2, S3 = close - (high-low)*1.25/2, S4 = close - (high-low)*1.5/2
    # CPP = (high + low + close)/3
    camarilla_r1 = prev_close_4h + (range_4h * 1.05 / 2)
    camarilla_s1 = prev_close_4h - (range_4h * 1.05 / 2)
    camarilla_cpp = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    
    # Get 12h data for EMA filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = close_12h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 4h Camarilla to 4h timeframe (no alignment needed for same timeframe)
    camarilla_r1_aligned = camarilla_r1
    camarilla_s1_aligned = camarilla_s1
    camarilla_cpp_aligned = camarilla_cpp
    
    # Align 12h EMAs to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_cpp_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(ema89_12h_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        cpp = camarilla_cpp_aligned[i]
        ema34 = ema34_12h_aligned[i]
        ema89 = ema89_12h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.3x avg AND 12h EMA34 > EMA89 (uptrend)
            if price > r1 and vol > 1.3 * vol_ma and ema34 > ema89:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.3x avg AND 12h EMA34 < EMA89 (downtrend)
            elif price < s1 and vol > 1.3 * vol_ma and ema34 < ema89:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla CPP OR 12h EMA34 < EMA89 (trend flip)
            if price < cpp or ema34 < ema89:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla CPP OR 12h EMA34 > EMA89 (trend flip)
            if price > cpp or ema34 > ema89:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CamarillaR1S1_Volume_12hEMA_Filter"
timeframe = "4h"
leverage = 1.0