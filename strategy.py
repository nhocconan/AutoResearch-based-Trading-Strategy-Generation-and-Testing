#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with HMA(21) trend filter and volume confirmation
# Uses 4h HMA21 for trend direction and Donchian channels from 4h for breakout signals
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe
# Works in both bull and bear markets by following the 4h trend direction
# Prioritizes BTC/ETH performance with SOL as secondary

name = "4h_Donchian20_HMA21_Trend_Volume"
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
    
    # Get 4h data for HMA calculation and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h HMA21 for trend filter
    close_4h = df_4h['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    half_n = 21 // 2
    wma_half = wma(close_4h, half_n)
    wma_full = wma(close_4h, 21)
    wma_2x_sub = 2 * wma_half - wma_full
    sqrt_n = int(np.sqrt(21))
    hma_21 = wma(wma_2x_sub, sqrt_n)
    # Pad to match original length
    hma_21_padded = np.full_like(close_4h, np.nan)
    hma_21_padded[half_n + sqrt_n - 1:] = hma_21[:len(close_4h) - half_n - sqrt_n + 1]
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21_padded)
    
    # Calculate Donchian channels (20-period) from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian Upper = max(high, 20)
    # Donchian Lower = min(low, 20)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(hma_21_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Donchian breakout with 4h HMA trend filter
        # Long: Price breaks above Donchian Upper (20) + volume spike + price above HMA21 (uptrend)
        # Short: Price breaks below Donchian Lower (20) + volume spike + price below HMA21 (downtrend)
        if position == 0:
            if (close[i] > upper_20_aligned[i] and volume_spike and 
                close[i] > hma_21_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lower_20_aligned[i] and volume_spike and 
                  close[i] < hma_21_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian Lower (20) OR price below HMA21 (trend change)
            if close[i] < lower_20_aligned[i] or close[i] < hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian Upper (20) OR price above HMA21 (trend change)
            if close[i] > upper_20_aligned[i] or close[i] > hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals