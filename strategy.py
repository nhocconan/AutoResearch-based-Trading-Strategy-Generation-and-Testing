#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
# Uses 4h primary timeframe for Donchian breakout signals (proven edge on SOLUSDT)
# 1d HMA confirms higher-timeframe trend direction to avoid counter-trend trades
# Volume spike (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.30) balances profit potential with fee drag minimization
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides clear structure, HMA filters false breakouts, volume confirms strength
# Works in both bull and bear markets by only trading breakouts aligned with 1d trend

name = "4h_Donchian20_1dHMA21_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA(21)
    def calculate_hma(arr, period):
        """Hull Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA of half period
        weights_half = np.arange(1, half_period + 1)
        wma_half = np.convolve(arr, weights_half/weights_half.sum(), mode='valid')
        # WMA of full period
        weights_full = np.arange(1, period + 1)
        wma_full = np.convolve(arr, weights_full/weights_full.sum(), mode='valid')
        # Raw HMA
        raw_hma = 2 * wma_half - wma_full
        # Final WMA of raw HMA
        weights_sqrt = np.arange(1, sqrt_period + 1)
        hma = np.convolve(raw_hma, weights_sqrt/weights_sqrt.sum(), mode='valid')
        
        # Align to original array
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1:period-1+len(hma)] = hma
        return result
    
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 4h Donchian channels (20-period)
    def calculate_donchian(high_arr, low_arr, period):
        """Donchian Channels"""
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = calculate_donchian(high, low, 20)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and HMA calculations)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian + 1d HMA uptrend + volume spike
            long_breakout = close[i] > upper_20[i]
            long_trend = close[i] > hma_21_1d_aligned[i]  # price above 1d HMA = uptrend
            
            # Short: price breaks below lower Donchian + 1d HMA downtrend + volume spike
            short_breakout = close[i] < lower_20[i]
            short_trend = close[i] < hma_21_1d_aligned[i]  # price below 1d HMA = downtrend
            
            if long_breakout and long_trend and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            elif short_breakout and short_trend and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend reverses
            if close[i] < lower_20[i] or close[i] < hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend reverses
            if close[i] > upper_20[i] or close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals