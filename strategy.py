#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w Donchian channel breakout and volume confirmation.
# Long: Price closes above 20-period weekly Donchian high + volume > 1.5x average daily volume (20-day).
# Short: Price closes below 20-period weekly Donchian low + volume > 1.5x average daily volume.
# Exit: Price crosses back below/above the weekly Donchian midpoint.
# Uses weekly structure for trend context, daily for execution with volume filter.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Works in bull (breakouts) and bear (mean reversion from extremes via exits).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 20-period Donchian channel (weekly)
    donchian_high = np.full(len(close_1w), np.nan)
    donchian_low = np.full(len(close_1w), np.nan)
    donchian_mid = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Average volume (20-period daily) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align weekly Donchian levels to daily
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        dm = donchian_mid_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price closes above weekly Donchian high + volume confirmation
            if (price > dh and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price closes below weekly Donchian low + volume confirmation
            elif (price < dl and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below weekly Donchian midpoint
            if price < dm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above weekly Donchian midpoint
            if price > dm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0