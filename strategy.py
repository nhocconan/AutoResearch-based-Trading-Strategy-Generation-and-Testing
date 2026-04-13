#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation (>1.8x 20-bar avg)
    # Enter long on breakout above Donchian high when 1d HMA(21) is rising, short on breakout below Donchian low when 1d HMA(21) is falling
    # Exit when price crosses Donchian midpoint
    # Volume filter requires significant participation (>1.8x average) to avoid false breakouts
    # HMA(21) on 1d provides smooth trend direction to align with higher timeframe momentum
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
    # Using 1d HTF for HMA reduces noise vs lower timeframes and improves trend alignment.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 1d data for HMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Hull Moving Average (HMA) 21-period on 1d
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    half_window = 21 // 2
    sqrt_window = int(np.sqrt(21))
    
    wma_half = wma(close_1d, half_window)
    wma_full = wma(close_1d, 21)
    wma_diff = 2 * wma_half - wma_full
    hma_21_1d = wma(wma_diff, sqrt_window)
    
    # Calculate HMA slope (trend direction): rising if current > previous
    hma_slope = np.diff(hma_21_1d, prepend=hma_21_1d[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Align 1d HMA and slope to 4h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    # Volume confirmation: volume > 1.8x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with HMA trend filter and volume confirmation
        long_entry = breakout_up and hma_rising_aligned[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_down and hma_falling_aligned[i] and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid_aligned[i])
        exit_short = (position == -1 and close[i] > donchian_mid_aligned[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "4h_1d_donchian_hma_volume_filter_v1"
timeframe = "4h"
leverage = 1.0