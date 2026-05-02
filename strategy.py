#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume spike
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Donchian(20) provides clear breakout structure with proven effectiveness on SOL/ETH
# 1d HMA(21) determines long-term trend bias with reduced lag vs EMA: long when price > HMA, short when price < HMA
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "4h_Donchian20_1dHMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(arr, half_period)
    wma_full = wma(arr, period)
    
    # Handle edge cases for array alignment
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad to original length
    result = np.full_like(arr, np.nan)
    start_idx = len(arr) - len(hma)
    if start_idx >= 0:
        result[start_idx:] = hma
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) channels (prior completed 4h bar's range)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d HMA(21) trend (prior completed 1d bar's HMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need at least 21 periods for HMA21
        return np.zeros(n)
    
    hma_21 = calculate_hma(df_1d['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Calculate 4h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 21)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d HMA21 (bullish bias) AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < 1d HMA21 (bearish bias) AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < hma_21_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below 1d HMA21 (trend change)
            if close[i] < low_ma[i] or close[i] < hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above 1d HMA21 (trend change)
            if close[i] > high_ma[i] or close[i] > hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals