#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dHMA_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts aligned with 1d HMA(34) trend filter and volume confirmation capture strong trends while avoiding whipsaws. HMA provides smooth, low-lag trend direction. Volume ensures breakout conviction. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Load 1d data ONCE before loop for HTF trend filter (HMA) and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA(34) for trend filter
    close_1d = df_1d['close'].values
    half_length = 34 // 2
    sqrt_length = int(np.sqrt(34))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    hma = np.full_like(close_1d, np.nan)
    for i in range(34 - 1, len(close_1d)):
        wma_half = wma(close_1d[i - half_length + 1:i + 1], half_length)
        wma_full = wma(close_1d[i - 34 + 1:i + 1], 34)
        hma[i] = 2 * wma_half - wma_full
    # Apply second WMA with sqrt length
    hma_final = np.full_like(close_1d, np.nan)
    for i in range(sqrt_length - 1, len(hma)):
        if not np.isnan(hma[i - sqrt_length + 1:i + 1]).any():
            hma_final[i] = wma(hma[i - sqrt_length + 1:i + 1], sqrt_length)
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_final)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])  # previous 1d close
    
    camarilla_range = high_1d - low_1d
    r1 = close_1d_shifted + 1.1 * camarilla_range / 12
    s1 = close_1d_shifted - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for HMA and volume MA)
    start_idx = max(40, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(hma_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (HMA)
        uptrend = close[i] > hma_aligned[i]
        downtrend = close[i] < hma_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume
        if uptrend and volume_spike and breakout_r1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S1 in downtrend with volume
        elif downtrend and volume_spike and breakout_s1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dHMA_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0