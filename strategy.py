#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation
    # Long: price > Donchian high(20) AND 1d HMA(21) rising AND volume > 1.5x 20-bar avg
    # Short: price < Donchian low(20) AND 1d HMA(21) falling AND volume > 1.5x 20-bar avg
    # Exit: price crosses Donchian midpoint OR volume dry-up
    # Using 4h primary for balance of signal quality and trade frequency,
    # Donchian for structure, 1d HMA for trend filter, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA(21)
    def calculate_hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        raw_hma = 2 * wma_half - wma_full
        hma = wma(raw_hma, sqrt_period)
        
        # Pad to original length
        result = np.full(len(arr), np.nan)
        result[period-1:] = hma
        return result
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Donchian midpoint
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: >1.5x 20-bar average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: HMA direction
        hma_rising = hma_1d_aligned[i] > hma_1d_aligned[i-1]
        hma_falling = hma_1d_aligned[i] < hma_1d_aligned[i-1]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + trend filter + volume confirmation
        long_entry = (close[i] > highest_high[i]) and hma_rising and vol_confirm
        short_entry = (close[i] < lowest_low[i]) and hma_falling and vol_confirm
        
        # Exit logic: price crosses Donchian midpoint OR volume dry-up
        long_exit = (close[i] < donchian_mid[i]) or not vol_confirm
        short_exit = (close[i] > donchian_mid[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0