#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation
    # Long: price > Donchian(20) high + price > 1d HMA(21) + volume > 1.5x avg volume
    # Short: price < Donchian(20) low + price < 1d HMA(21) + volume > 1.5x avg volume
    # Exit: price crosses Donchian midpoint (mean reversion)
    # Uses 1d HMA for trend alignment to avoid counter-trend trades
    # Donchian breakouts capture strong momentum moves
    # Volume confirmation filters weak breakouts
    # HMA filter ensures trading with higher timeframe trend
    # Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for HMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d HMA(21) for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = int(21 / 2)
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Pad arrays for WMA calculation
    def calculate_hma(close_vals, window):
        if len(close_vals) < window:
            return np.full_like(close_vals, np.nan)
        half_wma = np.full_like(close_vals, np.nan)
        full_wma = np.full_like(close_vals, np.nan)
        
        for i in range(window - 1, len(close_vals)):
            half_wma[i] = wma(close_vals[i - half_len + 1:i + 1], half_len)
            full_wma[i] = wma(close_vals[i - window + 1:i + 1], window)
        
        raw_hma = 2 * half_wma - full_wma
        hma = np.full_like(close_vals, np.nan)
        for i in range(sqrt_len - 1, len(close_vals)):
            hma[i] = wma(raw_hma[i - sqrt_len + 1:i + 1], sqrt_len)
        return hma
    
    hma_21_1d = calculate_hma(close_1d, 21)
    
    # Align 1d HMA21 to 12h timeframe
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate average volume (20-period) for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Trend filter from 1d HMA21
        uptrend = close[i] > hma_21_1d_aligned[i]
        downtrend = close[i] < hma_21_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Exit conditions
        exit_long = position == 1 and close[i] < donchian_mid[i]
        exit_short = position == -1 and close[i] > donchian_mid[i]
        
        # Entry conditions
        long_entry = breakout_up and uptrend and volume_confirm and position != 1
        short_entry = breakout_down and downtrend and volume_confirm and position != -1
        
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

name = "12h_1d_donchian_hma_volume_filter_v1"
timeframe = "12h"
leverage = 1.0