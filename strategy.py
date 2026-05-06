#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot breakouts with volume confirmation and choppiness regime filter
# Long when price breaks above daily R3 AND volume > 1.5 * avg_volume(20) AND choppiness < 61.8 (trending regime)
# Short when price breaks below daily S3 AND volume > 1.5 * avg_volume(20) AND choppiness < 61.8 (trending regime)
# Exit when price returns to daily pivot level or opposite Camarilla extreme (R4/S4)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Daily Camarilla pivots provide strong support/resistance levels from institutional order flow
# Volume confirmation filters out low-conviction breakouts
# Choppiness regime filter ensures we only trade in trending markets (avoids choppy whipsaws)
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "12h_1dCamarilla_R3S3_Breakout_Volume_ChopFilter"
timeframe = "12h"
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
    r4_1d = pivot_1d + (range_1d * 1.500)  # R4 = Pivot + 1.5 * range
    r3_1d = pivot_1d + (range_1d * 1.250)  # R3 = Pivot + 1.25 * range
    s3_1d = pivot_1d - (range_1d * 1.250)  # S3 = Pivot - 1.25 * range
    s4_1d = pivot_1d - (range_1d * 1.500)  # S4 = Pivot - 1.5 * range
    pp_1d = pivot_1d  # Pivot point for exit
    
    # Align daily Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate choppiness regime filter: CHOP(14) < 61.8 = trending regime
    # CHOP = 100 * LOG10(SUM(ATR(1),14) / (LOG10(MAX(HIGH,14)-MIN(LOW,14)) * SQRT(14)))
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr_1_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(max_high_14 - min_low_14) * np.sqrt(14)
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(sum_atr_1_14 / chop_denominator)
    chop_filter = chop < 61.8  # Trending regime when CHOP < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(pp_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or
            np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R3, volume spike, trending regime
            if (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1] and 
                volume_confirm[i] and chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S3, volume spike, trending regime
            elif (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1] and 
                  volume_confirm[i] and chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily pivot level or below daily S4
            if close[i] <= pp_1d_aligned[i] or close[i] >= s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily pivot level or above daily R4
            if close[i] >= pp_1d_aligned[i] or close[i] <= r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals