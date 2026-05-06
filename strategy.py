#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 1h volume spike and 12h HMA trend filter
# Long when price breaks above 12h Donchian upper band AND 1h volume > 2.0 * avg_volume(20) AND price > 12h HMA(21)
# Short when price breaks below 12h Donchian lower band AND 1h volume > 2.0 * avg_volume(20) AND price < 12h HMA(21)
# Exit when price crosses 12h Donchian middle band (mean reversion)
# Uses discrete sizing 0.30 to balance return and drawdown
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# 12h Donchian provides swing structure with proven breakout edge
# Volume spike confirms participation (reduces false breakouts)
# HMA trend filter ensures trades follow the higher timeframe momentum

name = "4h_12hDonchian20_1hVolumeSpike_HMATrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels and HMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need sufficient data for Donchian and HMA
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) channels based on previous 12h bar
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper_12h = rolling_max(high_12h, 20)
    donchian_lower_12h = rolling_min(low_12h, 20)
    donchian_middle_12h = (donchian_upper_12h + donchian_lower_12h) / 2.0
    
    # Calculate 12h HMA(21) for trend filter
    def wma(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan, dtype=float)
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, window):
        half_len = window // 2
        sqrt_len = int(np.sqrt(window))
        if half_len == 0:
            return np.full_like(arr, np.nan, dtype=float)
        wma_half = wma(arr, half_len)
        wma_full = wma(arr, window)
        if len(wma_half) < half_len or len(wma_full) < 1:
            return np.full_like(arr, np.nan, dtype=float)
        wma_half_extended = np.full_like(arr, np.nan, dtype=float)
        wma_half_extended[half_len-1:half_len-1+len(wma_half)] = wma_half
        raw_hma = 2 * wma_half_extended - wma_full
        hma_result = wma(raw_hma, sqrt_len)
        hma_final = np.full_like(arr, np.nan, dtype=float)
        hma_final[sqrt_len-1:sqrt_len-1+len(hma_result)] = hma_result
        return hma_final
    
    hma_21_12h = hma(close_12h, 21)
    
    # Get 1h data ONCE before loop for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:  # Need sufficient data for volume average
        return np.zeros(n)
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume_1h > (2.0 * avg_volume_20_1h)
    
    # Align 12h indicators to 4h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_12h)
    hma_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Align 1h indicators to 4h timeframe (wait for completed 1h bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1h, volume_spike_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(hma_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper with volume spike and HMA uptrend
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and close[i] > hma_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 12h Donchian lower with volume spike and HMA downtrend
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and close[i] < hma_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h Donchian middle (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 12h Donchian middle (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals