#!/usr/bin/env python3
"""
4h_PriceChannel_VolumeRegime
Strategy: 4h Donchian(20) breakout + volume confirmation + Chop regime filter.
Long: Price breaks above Donchian upper + volume > 1.5x 20-bar avg + Chop > 61.8 (range)
Short: Price breaks below Donchian lower + volume > 1.5x 20-bar avg + Chop > 61.8 (range)
Exit: Price crosses back through Donchian midpoint (mean reversion in chop)
Position size: 0.25
Uses 4h for structure, volume for confirmation, Chop for regime (avoid trending chop failures).
Works in bull/bear: Chop filter avoids whipsaws in strong trends, Donchian provides clear levels.
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
    
    # Get 4h data for Donchian channels and Chop
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) channels on 4h
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate Chop(14) on 4h: 100 * log15(sum(ATR(1)) / (max(high)-min(low))) / log15(14)
    atr_1 = np.maximum(np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1))), np.abs(low_4h - np.roll(close_4h, 1)))
    atr_1[0] = high_4h[0] - low_4h[0]  # first value
    sum_atr = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    roll_max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    roll_min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    range_14 = roll_max_high - roll_min_low
    chop = 100 * (np.log10(sum_atr) - np.log10(range_14)) / np.log10(14)
    chop = np.where(range_14 > 0, chop, 50)  # avoid division by zero
    
    # Align to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Get 4h volume for confirmation
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(20, n):  # warmup for Donchian(20)
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume aligned to 1h
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma20_4h_aligned[i])
        chop_filter = chop_aligned[i] > 61.8  # range regime
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        # Exit conditions: price crosses mid-line
        exit_long = close[i] < donchian_mid_aligned[i]
        exit_short = close[i] > donchian_mid_aligned[i]
        
        if position == 0:
            # Long: breakout up + volume + chop (range)
            if breakout_up and volume_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + volume + chop (range)
            elif breakout_down and volume_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below midpoint
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midpoint
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceChannel_VolumeRegime"
timeframe = "4h"
leverage = 1.0