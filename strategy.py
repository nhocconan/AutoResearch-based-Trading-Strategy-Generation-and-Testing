#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_v1
Hypothesis: Trade 4h Donchian(20) breakouts with volume confirmation and choppiness regime filter.
In trending markets (CHOP < 38.2): breakout continuation. In ranging markets (CHOP > 61.8): fade breakouts.
Position size: 0.25. Target: 80-150 total trades over 4 years (20-38/year).
Uses 1d HTF for trend alignment via EMA50 to avoid counter-trend trades in strong trends.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Choppiness Index (14-period) for regime filter
    atr_14 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_high_14 - lowest_low_14) * np.sqrt(14)
    chop = 100 * np.log10(atr_sum / chop_denom) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), volume MA (20), ATR (14), CHOP (14)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above 1d EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: CHOP < 38.2 = trending (breakout continuation), CHOP > 61.8 = ranging (fade breakouts)
        chop_value = chop[i]
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        if position == 0:
            # Long setup: price breaks above Donchian upper + volume + regime alignment
            long_breakout = close[i] > highest_20[i]
            long_setup = long_breakout and volume_confirm and (
                (is_trending and htf_1d_bullish) or  # In trend: only long if HTF bullish
                (is_ranging and not htf_1d_bullish)   # In range: fade (long if HTF not bullish)
            )
            
            # Short setup: price breaks below Donchian lower + volume + regime alignment
            short_breakout = close[i] < lowest_20[i]
            short_setup = short_breakout and volume_confirm and (
                (is_trending and htf_1d_bearish) or   # In trend: only short if HTF bearish
                (is_ranging and htf_1d_bullish)       # In range: fade (short if HTF bullish)
            )
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian lower OR 1d trend turns bearish in trending regime
            if (is_trending and close[i] <= lowest_20[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian upper OR 1d trend turns bullish in trending regime
            if (is_trending and close[i] >= highest_20[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0