#!/usr/bin/env python3
name = "4h_Vortex_VMP_Pos_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import sqrt
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex and VMP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Vortex Indicator (14-period standard)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    vm_plus[0] = np.abs(high_1d[0] - low_1d[-1])  # first value uses previous day's low
    vm_minus[0] = np.abs(low_1d[0] - high_1d[-1])  # first value uses previous day's high
    
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d, 1)),
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = np.maximum(high_1d[0] - low_1d[0], 
                        np.absolute(high_1d[0] - close_1d[-1]),
                        np.absolute(low_1d[0] - close_1d[-1]))
    
    atr14 = np.zeros_like(tr1)
    atr14[0] = tr1[0]
    for i in range(1, len(tr1)):
        atr14[i] = (atr14[i-1] * 13 + tr1[i]) / 14
    
    vi_plus = np.zeros_like(vm_plus)
    vi_minus = np.zeros_like(vm_minus)
    for i in range(14, len(vm_plus)):
        vi_plus[i] = np.sum(vm_plus[i-13:i+1]) / np.sum(atr14[i-13:i+1])
        vi_minus[i] = np.sum(vm_minus[i-13:i+1]) / np.sum(atr14[i-13:i+1])
    
    # VMP (Volume Moving Average Position) - volume relative to 20-period average
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20[i] = np.mean(df_1d['volume'].values[i-20:i])
    vmp = df_1d['volume'].values / vol_ma_20
    
    # Align indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    vmp_aligned = align_htf_to_ltf(prices, df_1d, vmp)
    
    # Volume filter: current volume > 1.8x 20-period average (on 4h)
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.8 * vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~12 hours for 4h to reduce trades
    
    start_idx = max(200, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or 
            np.isnan(vmp_aligned[i]) or 
            np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: VI+ > VI- (bullish vortex) + VMP > 1.2 (strong volume) + volume filter
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                vmp_aligned[i] > 1.2 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: VI- > VI+ (bearish vortex) + VMP > 1.2 (strong volume) + volume filter
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  vmp_aligned[i] > 1.2 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: VI- crosses above VI+ (vortex bearish crossover)
            if vi_minus_aligned[i] >= vi_plus_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: VI+ crosses above VI- (vortex bullish crossover)
            if vi_plus_aligned[i] >= vi_minus_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Vortex Indicator identifies trend direction (VI+ > VI- = bullish, VI- > VI+ = bearish)
# combined with Volume Moving Average Position (VMP) to confirm institutional participation.
# Long when bullish vortex + strong volume (VMP > 1.2). Short when bearish vortex + strong volume.
# Uses 4h timeframe with volume confirmation to filter false signals. Target: 80-150 total trades over 4 years.
# Works in both bull (strong volume on uptrends) and bear (strong volume on downtrends) markets.