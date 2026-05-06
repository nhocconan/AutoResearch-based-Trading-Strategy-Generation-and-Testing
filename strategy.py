#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla pivot levels from 1d with volume confirmation and chop regime filter
# Long when price touches or breaks above Camarilla R3 level AND chop regime indicates trending (CHOP < 38.2) AND volume spike
# Short when price touches or breaks below Camarilla S3 level AND chop regime indicates trending (CHOP < 38.2) AND volume spike
# Exit when price reaches Camarilla Pivot level (midpoint) or opposite S1/R1 level
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Camarilla levels from 1d provide strong intraday support/resistance that works in both bull and bear markets
# Chop filter ensures we only trade in trending conditions, avoiding whipsaws in ranging markets
# Volume confirmation ensures breakouts have conviction
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_1dChop_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed daily bars for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (high + low + close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = high - low
    range_1d = high_1d - low_1d
    # Camarilla levels
    r3_1d = pivot_1d + range_1d * 1.1 / 4.0
    r2_1d = pivot_1d + range_1d * 1.1 / 6.0
    r1_1d = pivot_1d + range_1d * 1.1 / 12.0
    s1_1d = pivot_1d - range_1d * 1.1 / 12.0
    s2_1d = pivot_1d - range_1d * 1.1 / 6.0
    s3_1d = pivot_1d - range_1d * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1d data for Chop regime filter (using 1d OHLC)
    # Chop = 100 * log10(sum(ATR(14)) / log(n) / (max(high,n) - min(low,n)))
    # Simplified: use 14-period chop on 1d
    if len(df_1d) < 14:
        chop_1d = np.full(len(df_1d), 50.0)  # neutral if not enough data
    else:
        # Calculate True Range for 1d
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])
        
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
        max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Avoid division by zero
        range_14 = max_high_14 - min_low_14
        range_14 = np.where(range_14 == 0, 1e-10, range_14)
        
        chop_1d = 100 * np.log10(sum_atr_14 / np.log(14) / range_14)
        chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when chop indicates trending (CHOP < 38.2)
        is_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: price touches/breaks above R3, trending regime, volume spike
            if (close[i] >= r3_1d_aligned[i] and close[i-1] < r3_1d_aligned[i-1] and 
                is_trending and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches/breaks below S3, trending regime, volume spike
            elif (close[i] <= s3_1d_aligned[i] and close[i-1] > s3_1d_aligned[i-1] and 
                  is_trending and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches pivot or S1 level
            if close[i] <= pivot_1d_aligned[i] or close[i] <= s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches pivot or R1 level
            if close[i] >= pivot_1d_aligned[i] or close[i] >= r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals