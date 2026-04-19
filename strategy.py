#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and 1d trend filter
# Uses tight entry conditions (S1/S3 levels) to limit trades and avoid fee drag
# Works in bull markets via long at S1 in uptrend and short at S3 in downtrend
# Only trades when volume confirms and higher timeframe trend aligns
name = "4h_CamarillaPivot_Reversal_VolumeTrend_v1"
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
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's Camarilla levels
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    range_prev = high_prev - low_prev
    # S3 = C - (H-L)*1.1/2
    s3 = close_prev - range_prev * 1.1 / 2
    # S2 = C - (H-L)*1.1/4
    s2 = close_prev - range_prev * 1.1 / 4
    # S1 = C - (H-L)*1.1/6
    s1 = close_prev - range_prev * 1.1 / 6
    # R1 = C + (H-L)*1.1/6
    r1 = close_prev + range_prev * 1.1 / 6
    # R2 = C + (H-L)*1.1/4
    r2 = close_prev + range_prev * 1.1 / 4
    # R3 = C + (H-L)*1.1/2
    r3 = close_prev + range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: price touches S1 in uptrend with volume confirmation
            if low[i] <= s1_aligned[i] and volume_filter and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches R3 in downtrend with volume confirmation
            elif high[i] >= r3_aligned[i] and volume_filter and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price touches S3 or R1 or ATR-based stop
            if low[i] <= s3_aligned[i] or high[i] >= r1_aligned[i] or close[i] < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price touches R3 or S1 or ATR-based stop
            if high[i] >= r3_aligned[i] or low[i] <= s1_aligned[i] or close[i] > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals