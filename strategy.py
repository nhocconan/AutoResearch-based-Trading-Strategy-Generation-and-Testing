#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
Long when close breaks above Donchian upper band AND 1w HMA rising AND volume > 2.0x 20-period MA.
Short when close breaks below Donchian lower band AND 1w HMA falling AND volume > 2.0x 20-period MA.
Exit when price crosses Donchian middle band (20-period SMA) or trend reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades in bear markets like 2025.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Donchian channels provide structural breakouts, HMA reduces lag, volume confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    middle_band = (upper_band + lower_band) / 2.0  # 20-period SMA of midpoint
    
    # Calculate 1w HMA(21) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    wma_half = wma(close_1w, half_len)
    wma_full = wma(close_1w, 21)
    wma_2x_sub = 2 * wma_half - wma_full[-len(wma_half):] if len(wma_half) > 0 else np.array([])
    hma_21 = wma(wma_2x_sub, sqrt_len) if len(wma_2x_sub) >= sqrt_len else np.array([])
    
    # Pad HMA to match close_1w length
    hma_21_padded = np.full_like(close_1w, np.nan)
    if len(hma_21) > 0:
        start_idx = 21 - len(hma_21)  # Adjust for WMA padding
        end_idx = start_idx + len(hma_21)
        if start_idx >= 0 and end_idx <= len(close_1w):
            hma_21_padded[start_idx:end_idx] = hma_21
    
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
    
    # Calculate 1d volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20, 20)  # Donchian (20), HMA (needs 21+), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate HMA slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            hma_prev = hma_21_aligned[i-1]
            hma_rising = hma_21_aligned[i] > hma_prev
            hma_falling = hma_21_aligned[i] < hma_prev
        else:
            hma_rising = False
            hma_falling = False
        
        # Volume filter: 1d volume > 2.0x 20-period MA (high threshold to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: close breaks above upper band AND HMA rising AND volume filter
            if close[i] > upper_band[i] and hma_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below lower band AND HMA falling AND volume filter
            elif close[i] < lower_band[i] and hma_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below middle band OR HMA starts falling
                if close[i] < middle_band[i] or (i >= start_idx + 1 and hma_21_aligned[i] < hma_21_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above middle band OR HMA starts rising
                if close[i] > middle_band[i] or (i >= start_idx + 1 and hma_21_aligned[i] > hma_21_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wHMA21_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0