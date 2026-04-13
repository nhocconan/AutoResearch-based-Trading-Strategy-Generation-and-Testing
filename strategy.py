#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation
    # Long: price breaks above 4h Donchian upper channel + 1d HMA21 uptrend + volume > 1.5x 20-period average
    # Short: price breaks below 4h Donchian lower channel + 1d HMA21 downtrend + volume > 1.5x 20-period average
    # Exit: price crosses 4h Donchian midline (median of upper/lower)
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for HMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = highest_high_4h
    donchian_lower_4h = lowest_low_4h
    donchian_middle_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Calculate 1d HMA21 for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = int(21 / 2)
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights/weights.sum(), mode='same')
    
    wma_half_1d = wma(close_1d, half_len)
    wma_full_1d = wma(close_1d, 21)
    raw_hma_1d = 2 * wma_half_1d - wma_full_1d
    hma_21_1d = wma(raw_hma_1d, sqrt_len)
    
    # Align 1d HMA21 to 4h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 4h volume average (20-period)
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(volume_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper_4h[i-1]  # break above previous upper
        breakout_down = close[i] < donchian_lower_4h[i-1]  # break below previous lower
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma_4h[i]
        
        # Trend filter from 1d HMA21
        uptrend = close[i] > hma_21_1d_aligned[i]
        downtrend = close[i] < hma_21_1d_aligned[i]
        
        # Exit conditions: price crosses Donchian midline
        exit_long = position == 1 and close[i] < donchian_middle_4h[i]
        exit_short = position == -1 and close[i] > donchian_middle_4h[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_confirmed and uptrend and position != 1
        short_entry = breakout_down and volume_confirmed and downtrend and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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