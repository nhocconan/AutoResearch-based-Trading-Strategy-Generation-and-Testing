#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper band AND 12h HMA21 > 12h HMA21 previous bar (uptrend) AND volume > 1.5x 20-period MA.
Short when price breaks below 4h Donchian lower band AND 12h HMA21 < 12h HMA21 previous bar (downtrend) AND volume > 1.5x 20-period MA.
Exit when price retraces to the 4h Donchian middle band (20-period SMA) or 12h trend reverses.
Designed for low trade frequency (target: 25-40/year) with strong structure from proven Donchian patterns.
Volume filter set at 1.5x to reduce false breakouts while maintaining sufficient trades.
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
    
    # Calculate 4h Donchian channels (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = SMA of close over 20 periods
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    close_roll = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_middle = close_roll
    
    # Calculate 12h HMA(21) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='full')[:len(values)] * weights[::-1] / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21 = wma(raw_hma, sqrt_len)
    
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    hma_21_prev = np.roll(hma_21_aligned, 1)
    hma_21_prev[0] = np.nan
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21) + 1  # +1 for the roll(1) shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(hma_21_prev[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h HMA21 rising = uptrend, falling = downtrend
        trend_up = hma_21_aligned[i] > hma_21_prev[i]
        trend_down = hma_21_aligned[i] < hma_21_prev[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND uptrend AND volume filter
            if close[i] > donchian_upper[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend AND volume filter
            elif close[i] < donchian_lower[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retraces to Donchian middle band OR 12h trend turns down
                if close[i] <= donchian_middle[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retraces to Donchian middle band OR 12h trend turns up
                if close[i] >= donchian_middle[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hHMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0