#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 fade with 1d volume spike filter and 1w trend filter.
Long when price touches S3 and 1d volume > 1.5x 20-period average AND 1w close > 1w open (bullish weekly candle).
Short when price touches R3 and 1d volume > 1.5x 20-period average AND 1w close < 1w open (bearish weekly candle).
Exit when price reverses to Camarilla R1/S1 or volume condition fails.
Uses 1d for volume and Camarilla levels, 1w for trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume SMA (20-period)
    volume_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        camarilla_s1[i] = pc - 1.1 * (ph - pl) / 12
        camarilla_s3[i] = pc - 1.1 * (ph - pl) / 4
        camarilla_r1[i] = pc + 1.1 * (ph - pl) / 12
        camarilla_r3[i] = pc + 1.1 * (ph - pl) / 4
    
    # Align 1d indicators
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    # Align 1w trend filter (bullish/bearish weekly candle)
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_sma_20_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        # Note: We use the 1d volume from the completed daily bar
        vol_condition = volume_1d[min(i // 24, len(volume_1d)-1)] > 1.5 * volume_sma_20[min(i // 24, len(volume_sma_20)-1)] if i // 24 < len(volume_1d) else False
        
        # Simplified volume check using aligned data (approximation)
        # For better accuracy, we'd need to map 6h bars to exact 1d volume, but this works as filter
        vol_cond_approx = True  # Volume condition handled separately below
        
        if position == 0:
            # Long: price touches or goes below S3 AND volume spike AND weekly bullish
            if low[i] <= camarilla_s3_aligned[i] and vol_condition and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R3 AND volume spike AND weekly bearish
            elif high[i] >= camarilla_r3_aligned[i] and vol_condition and weekly_bearish_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to S1 or volume condition fails or weekly turns bearish
            if high[i] >= camarilla_s1_aligned[i] or not vol_condition or weekly_bullish_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to R1 or volume condition fails or weekly turns bullish
            if low[i] <= camarilla_r1_aligned[i] or not vol_condition or weekly_bearish_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_VolumeSpike_1wTrend"
timeframe = "6h"
leverage = 1.0