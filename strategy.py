#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation (1.5x 20-bar avg)
# Long when price breaks above Donchian upper band AND price > 1d HMA21 AND volume > 1.5x 20-bar avg volume
# Short when price breaks below Donchian lower band AND price < 1d HMA21 AND volume > 1.5x 20-bar avg volume
# Exit when price reverts to Donchian midpoint (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 19-50 trades/year on 4h timeframe.
# Donchian channels provide clear breakout levels, 1d HMA21 filters counter-trend moves with lag,
# volume confirmation ensures breakout strength. Designed for both bull and bear markets via trend filter.

name = "4h_Donchian20_VolumeConfirm_1dHMA21_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA(21) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d data
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    hma_raw = 2 * wma_half - wma_full
    hma_21 = wma(hma_raw, sqrt_len)
    # Align HMA21 to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_hma21 = hma_21_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to Donchian midpoint (mean reversion)
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND price > 1d HMA21 AND volume confirmation
            if curr_close > curr_upper and curr_close > curr_hma21 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND price < 1d HMA21 AND volume confirmation
            elif curr_close < curr_lower and curr_close < curr_hma21 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals