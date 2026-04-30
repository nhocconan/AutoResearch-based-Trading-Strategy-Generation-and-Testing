#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# Donchian channel provides clear breakout levels; HMA confirms medium-term trend direction;
# Volume spike validates breakout strength. Works in bull via upside breakouts, in bear via downside breakouts.
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_12hHMA21_VolumeSpike_v1"
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
    
    # Calculate 4h Donchian(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h HMA(21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    hma_21 = calculate_hma(df_12h['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # warmup for Donchian and HMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_hma = hma_21_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above Donchian high AND above HMA
                if curr_close > curr_donchian_high and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below Donchian low AND below HMA
                elif curr_close < curr_donchian_low and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price drops below Donchian low
            if curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high
            if curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(close)
    for i in range(len(close)):
        if i < half_period - 1:
            wma_half[i] = np.nan
        else:
            start_idx = i - half_period + 1
            weights = np.arange(1, half_period + 1)
            wma_half[i] = np.dot(close[start_idx:i+1], weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.zeros_like(close)
    for i in range(len(close)):
        if i < period - 1:
            wma_full[i] = np.nan
        else:
            start_idx = i - period + 1
            weights = np.arange(1, period + 1)
            wma_full[i] = np.dot(close[start_idx:i+1], weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = np.zeros_like(close)
    for i in range(len(close)):
        if i < sqrt_period - 1:
            hma[i] = np.nan
        else:
            start_idx = i - sqrt_period + 1
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.dot(raw_hma[start_idx:i+1], weights) / weights.sum()
    
    return hma