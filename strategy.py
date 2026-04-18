#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_WithVolume_1dTrend
Hypothesis: Camarilla pivot breakouts at R1/S1 levels on 6h with volume confirmation and 1d EMA50 trend filter.
Long when price breaks above R1 with volume spike and above EMA50; short when breaks below S1 with volume spike and below EMA50.
Uses daily pivots calculated from previous day's OHLC. Designed for low trade frequency (15-30/year) to avoid fee drag.
Works in bull/bear markets by aligning with 1d trend - only takes long in uptrend, short in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # R2 = close + 0.5 * (high - low)
    # R1 = close + 0.25 * (high - low)
    # S1 = close - 0.25 * (high - low)
    # S2 = close - 0.5 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    high_low_range = prev_high - prev_low
    
    r1 = prev_close + 0.25 * high_low_range
    s1 = prev_close - 0.25 * high_low_range
    
    # Align daily pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: >1.8x 24-period average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 25)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_6h[i]
        s1_level = s1_6h[i]
        ema50 = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and uptrend
            if price > r1_level and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and downtrend
            elif price < s1_level and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below R1 OR trend turns down
            if price < r1_level or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above S1 OR trend turns up
            if price > s1_level or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_WithVolume_1dTrend"
timeframe = "6h"
leverage = 1.0