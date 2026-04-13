#!/usr/bin/env python3
"""
4h_1d_DonchianBreakout_VolumeTrend_v2
Hypothesis: Price breaking above/below 4-hour Donchian channels (20-period) with daily volume confirmation and trend filter captures momentum moves. Uses higher timeframe (1d) for volume and trend context to filter false breakouts. Designed for low trade frequency (<400 total) to minimize fee drag in both bull and bear markets.
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
    
    # Get daily data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4-hour Donchian channels (20-period)
    # Using 4h high/low directly from prices
    high_4h = high
    low_4h = low
    
    # Rolling max/min for upper/lower bands
    upper_donchian = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume_1d > (vol_ma_20 * 1.5)
    
    # Daily trend filter: price above/below 50-day SMA
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    price_above_sma = close_1d > sma_50
    price_below_sma = close_1d < sma_50
    
    # Align all daily signals to 4h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed.astype(float))
    price_above_sma_aligned = align_htf_to_ltf(prices, df_1d, price_above_sma.astype(float))
    price_below_sma_aligned = align_htf_to_ltf(prices, df_1d, price_below_sma.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(volume_confirmed_aligned[i]) or 
            np.isnan(price_above_sma_aligned[i]) or 
            np.isnan(price_below_sma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Break of Donchian channels with daily volume and trend confirmation
        long_break = close[i] > upper_donchian_aligned[i]
        short_break = close[i] < lower_donchian_aligned[i]
        
        long_entry = long_break and volume_confirmed_aligned[i] > 0.5 and price_above_sma_aligned[i] > 0.5
        short_entry = short_break and volume_confirmed_aligned[i] > 0.5 and price_below_sma_aligned[i] > 0.5
        
        # Exit when price returns to opposite Donchian band (mean reversion)
        exit_long = position == 1 and close[i] <= lower_donchian_aligned[i]
        exit_short = position == -1 and close[i] >= upper_donchian_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
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

name = "4h_1d_DonchianBreakout_VolumeTrend_v2"
timeframe = "4h"
leverage = 1.0