#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
    # Long when price breaks above Donchian upper band AND 12h HMA rising AND volume > 1.5x avg.
    # Short when price breaks below Donchian lower band AND 12h HMA falling AND volume > 1.5x avg.
    # Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.
    # Target: 80-160 total trades over 4 years (20-40/year) to balance alpha and fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights, mode='valid') / weights.sum()
    
    def hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        # Align arrays: wma_half starts at index half_period-1, wma_full at period-1
        raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
        hma_vals = wma(raw_hma, sqrt_period)
        # Pad to original length
        result = np.full_like(arr, np.nan)
        result[period-1:] = hma_vals
        return result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above previous upper band
        short_breakout = close[i] < lowest_low[i-1]   # Break below previous lower band
        
        # HMA trend filter: rising for long, falling for short
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        
        # Entry conditions
        long_entry = long_breakout and volume_confirmed and hma_rising
        short_entry = short_breakout and volume_confirmed and hma_falling
        
        # Exit conditions: Donchian middle band or opposite breakout
        middle_band = (highest_high[i] + lowest_low[i]) / 2
        long_exit = close[i] < middle_band
        short_exit = close[i] > middle_band
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry logic
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit logic
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_hma_volume_filter_v1"
timeframe = "4h"
leverage = 1.0