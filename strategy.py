#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation.
Long when price breaks above Donchian upper band and 1d HMA(21) is rising, with volume > 1.5x average.
Short when price breaks below Donchian lower band and 1d HMA(21) is falling, with volume > 1.5x average.
Exit when price returns to Donchian middle band (mean reversion) or volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 4h timeframe to capture
medium-term trends while avoiding overtrading. Works in both bull and bear markets by requiring
trend confirmation via 1d HMA direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HMA(21) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) for 1d timeframe
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Pad arrays for convolution
    wma_half = np.full_like(close_1d, np.nan)
    wma_full = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= half_len:
        wma_half[half_len-1:] = wma(close_1d, half_len)
    if len(close_1d) >= 21:
        wma_full[20:] = wma(close_1d, 21)
    
    wma_2x_half = 2 * wma_half
    wma_diff = wma_2x_half - wma_full
    
    hma_1d = np.full_like(close_1d, np.nan)
    if len(wma_diff) >= sqrt_len:
        hma_values = wma(wma_diff[~np.isnan(wma_diff)], sqrt_len)
        hma_1d[sqrt_len-1:sqrt_len-1+len(hma_values)] = hma_values
    
    # Calculate HMA direction (rising/falling)
    hma_rising = np.full_like(close_1d, False)
    hma_falling = np.full_like(close_1d, False)
    for i in range(1, len(hma_1d)):
        if not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-1]):
            if hma_1d[i] > hma_1d[i-1]:
                hma_rising[i] = True
            elif hma_1d[i] < hma_1d[i-1]:
                hma_falling[i] = True
    
    # Align HTF indicators to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(vol_ma[i]) or
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hma_rise = hma_rising_aligned[i]
        hma_fall = hma_falling_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band, HMA rising, volume confirmation
            if (close[i] > donchian_high[i] and hma_rise and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, HMA falling, volume confirmation
            elif (close[i] < donchian_low[i] and hma_fall and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to Donchian middle band OR volume drops below average
                if close[i] <= donchian_middle[i] or vol_current < vol_ma_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to Donchian middle band OR volume drops below average
                if close[i] >= donchian_middle[i] or vol_current < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dHMA_Volume_Breakout"
timeframe = "4h"
leverage = 1.0