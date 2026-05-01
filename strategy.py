#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Weekly Camarilla pivots provide institutional reference points; breakout in direction of weekly trend
# captures momentum with lower false signals. Volume spike confirms participation. Works in bull/bear
# by trading with weekly trend. Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.

name = "6h_Donchian20_1wCamarilla_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and ranges
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels (based on prior week)
    r4_1w = pivot_1w + range_1w * 1.1 / 2
    r3_1w = pivot_1w + range_1w * 1.1 / 4
    s3_1w = pivot_1w - range_1w * 1.1 / 4
    s4_1w = pivot_1w - range_1w * 1.1 / 2
    
    # Align weekly levels to 6h timeframe (completed weekly bar only)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Weekly trend: price above/below weekly pivot
    uptrend_1w = close > pivot_1w_aligned
    downtrend_1w = close < pivot_1w_aligned
    
    # Donchian(20) channels on 6h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 1)  # Need sufficient history for Donchian and weekly data
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20[i-1]  # Price breaks above prior 20-period high
        breakout_down = close[i] < lowest_low_20[i-1]  # Price breaks below prior 20-period low
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout above R4, volume spike, weekly uptrend
            if breakout_up and vol_spike and uptrend_1w[i] and close[i] > r4_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below S4, volume spike, weekly downtrend
            elif breakout_down and vol_spike and downtrend_1w[i] and close[i] < s4_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or weekly trend reversal
            if close[i] < lowest_low_20[i] or not uptrend_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or weekly trend reversal
            if close[i] > highest_high_20[i] or not downtrend_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals