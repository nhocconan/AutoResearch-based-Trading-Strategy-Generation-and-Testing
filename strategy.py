#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout + 1d HMA21 trend filter + volume confirmation
# Long when price breaks above 4h Donchian upper band AND price > 1d HMA21 AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 4h Donchian lower band AND price < 1d HMA21 AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price crosses back below/above 4h Donchian middle band OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides robust breakout levels from price structure
# 1d HMA21 filters primary trend to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_Donchian20_1dHMA21_VolumeSpike"
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
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (based on previous 20-period)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    middle_4h = (upper_4h + lower_4h) / 2.0
    
    # Align 4h Donchian channels to 4h timeframe (already aligned, no shift needed)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Get 1d data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need enough for HMA21
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA21 (Hull Moving Average)
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    wma_raw = 2 * wma_half - wma_full
    hma_21 = wma(wma_raw, sqrt_len)
    
    # Align 1d HMA21 to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(middle_4h_aligned[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper band, above 1d HMA21, volume confirmation, in session
            if close[i] > upper_4h_aligned[i] and close[i] > hma_21_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 4h Donchian lower band, below 1d HMA21, volume confirmation, in session
            elif close[i] < lower_4h_aligned[i] and close[i] < hma_21_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian middle band OR volume drops below average
            if close[i] < middle_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian middle band OR volume drops below average
            if close[i] > middle_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals