#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout + 12h HMA trend + volume confirmation
    # Long: price > Donchian(20) high AND 12h HMA(21) rising AND volume > 1.5x avg
    # Short: price < Donchian(20) low AND 12h HMA(21) falling AND volume > 1.5x avg
    # Exit: price crosses Donchian midpoint OR volume drops below avg
    # Uses 4h for price action/volume, 12h for trend filter
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 100-200 total trades over 4 years (~25-50/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and volume (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 12h data for HMA (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian Channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian high and low
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 4h Donchian to 4h timeframe (no additional delay for price-based indicators)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate 4h volume average (20-period)
    volume_4h = df_4h['volume'].values
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    # Calculate 12h HMA (Hull Moving Average) for trend
    close_12h = df_12h['close'].values
    
    # HMA formula: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(data, weights/weights.sum(), mode='valid')
    
    def hma(data, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        wma_half = wma(data, half_period)
        wma_full = wma(data, period)
        
        # 2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full
        
        # WMA(sqrt(n)) of the above
        hma_val = wma(raw_hma, sqrt_period)
        
        # Pad to original length
        result = np.full_like(data, np.nan)
        result[period-1:period-1+len(hma_val)] = hma_val
        return result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # HMA slope (rising/falling)
    hma_slope = np.diff(hma_12h_aligned, prepend=hma_12h_aligned[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * volume_ma_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Exit conditions
        long_exit = close[i] < donchian_mid_aligned[i]
        short_exit = close[i] > donchian_mid_aligned[i]
        volume_exit = volume[i] < volume_ma_aligned[i]  # volume drops below average
        
        # Entry logic: breakout + trend + volume confirmation
        long_entry = long_breakout and hma_rising[i] and volume_confirm
        short_entry = short_breakout and hma_falling[i] and volume_confirm
        
        # Exit logic: midpoint cross OR volume drops
        long_exit_condition = long_exit or volume_exit
        short_exit_condition = short_exit or volume_exit
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit_condition:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit_condition:
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

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0