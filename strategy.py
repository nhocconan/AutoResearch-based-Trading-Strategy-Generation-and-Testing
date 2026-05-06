#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot breakouts with volume confirmation and chop regime filter
# Long when price breaks above daily R3 AND volume > 1.3 * avg_volume(20) AND chop > 61.8 (range)
# Short when price breaks below daily S3 AND volume > 1.3 * avg_volume(20) AND chop > 61.8 (range)
# Exit when price returns to daily midpoint (pivot level) or opposite extreme
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Daily Camarilla pivots provide strong intraday support/resistance levels
# Chop regime filter ensures we trade in ranging markets where mean reversion works
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (mean reversion in ranges) markets

name = "4h_1dCamarilla_R3S3_Breakout_Volume_Chop"
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
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = pivot_1d + (range_1d * 1.125)  # R3 = Pivot + 1.125 * range
    r4_1d = pivot_1d + (range_1d * 1.500)  # R4 = Pivot + 1.5 * range
    s3_1d = pivot_1d - (range_1d * 1.125)  # S3 = Pivot - 1.125 * range
    s4_1d = pivot_1d - (range_1d * 1.500)  # S4 = Pivot - 1.5 * range
    midpoint_1d = pivot_1d  # Camarilla midpoint is the pivot point
    
    # Align daily Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Calculate Choppiness Index regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging market (good for mean reversion)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (14 * np.log10(14))) / np.log10(14)
    chop_regime = chop > 61.8  # Ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(midpoint_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R3, volume spike, ranging market
            if (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1] and 
                volume_confirm[i] and chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S3, volume spike, ranging market
            elif (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1] and 
                  volume_confirm[i] and chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily midpoint or below
            if close[i] <= midpoint_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily midpoint or above
            if close[i] >= midpoint_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals