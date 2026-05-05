#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 Breakout with 1d HMA50 Trend Filter and Volume Spike
# Long when price breaks above R4 (1d) AND price > 1d HMA50 (strong uptrend) AND volume spike
# Short when price breaks below S4 (1d) AND price < 1d HMA50 (strong downtrend) AND volume spike
# R4/S4 are the strongest Camarilla levels (PP ± range/2) for very high-quality breaks
# HMA50 provides smooth trend filter with less lag than EMA/SMA
# Volume spike requires 2.5x 20-bar MA for confirmation (stricter than before)
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 4h (primary timeframe as required)

name = "4h_Camarilla_R4S4_Breakout_1dHMA50_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Camarilla and HMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA50
    close_1d = df_1d['close'].values
    half_len = 50 // 2
    sqrt_len = int(np.sqrt(50))
    wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    wma_sqrt = pd.Series(2 * wma_half - wma_full).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_50_1d = wma_sqrt
    hma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    
    # Calculate pivot point (PP) = (H+L+C)/3
    pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3.0
    # Calculate range
    range_1d = high_1d_shifted - low_1d_shifted
    # Camarilla levels (R4/S4 = strongest levels at PP ± range/2)
    r4 = pp + (range_1d / 2.0)  # R4 = PP + range/2
    s4 = pp - (range_1d / 2.0)  # S4 = PP - range/2
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation on 4h (stricter threshold: 2.5x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.5 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(hma_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 AND strong uptrend (price > HMA50) AND volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > hma_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND strong downtrend (price < HMA50) AND volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < hma_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R4 OR closes below HMA50
            if close[i] < r4_aligned[i] or close[i] < hma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S4 OR closes above HMA50
            if close[i] > s4_aligned[i] or close[i] > hma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals