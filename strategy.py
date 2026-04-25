#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + HMA(21) Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. 
HMA(21) on 4h filters trend direction to avoid counter-trend trades.
Volume spike confirms institutional participation. Works in bull/bear via HMA slope.
Target: 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(arr, weights_half, mode='valid') / weights_half.sum()
    wma_half = np.concatenate([np.full(half_period - 1, np.nan), wma_half])
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(arr, weights_full, mode='valid') / weights_full.sum()
    wma_full = np.concatenate([np.full(period - 1, np.nan), wma_full])
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    hma = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    hma = np.concatenate([np.full(sqrt_period - 1, np.nan), hma])
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate HMA(21) on 4h close
    hma_4h = calculate_hma(close, 21)
    hma_slope = np.diff(hma_4h, prepend=np.nan)  # slope = change from previous bar
    
    # Calculate Donchian(20) channels
    # Upper channel: highest high over past 20 periods
    # Lower channel: lowest low over past 20 periods
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-19:i+1])
        lower_channel[i] = np.min(low[i-19:i+1])
    
    # Volume spike: current volume > 2.0 * 20-period average
    volume_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        volume_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (2.0 * volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(hma_4h[i]) or 
            np.isnan(hma_slope[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Determine trend direction from HMA slope
        # Positive slope = uptrend, Negative slope = downtrend
        hma_up = hma_slope[i] > 0
        hma_down = hma_slope[i] < 0
        
        if position == 0:
            # Long: price breaks above upper channel AND HMA uptrend AND volume spike
            long_condition = (curr_close > upper_channel[i]) and hma_up and volume_spike[i]
            # Short: price breaks below lower channel AND HMA downtrend AND volume spike
            short_condition = (curr_close < lower_channel[i]) and hma_down and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below lower channel or HMA turns down
            if curr_close < lower_channel[i] or hma_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above upper channel or HMA turns up
            if curr_close > upper_channel[i] or hma_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_HMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0