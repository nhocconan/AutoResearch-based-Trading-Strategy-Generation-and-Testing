#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d Williams %R filter and volume confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d Williams %R(14) for momentum (oversold < -80 for long, overbought > -20 for short).
- Volume: Current 6h volume > 1.8 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above Donchian(20) high AND Williams %R < -80 AND volume spike.
         Short when price breaks below Donchian(20) low AND Williams %R > -20 AND volume spike.
- Exit: Opposite Donchian level (low for long, high for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Donchian channels provide objective breakout levels. Williams %R filters for momentum extremes,
avoiding breakouts in choppy markets. Volume confirmation ensures institutional participation.
Works in both bull and bear markets by only taking breakouts aligned with 1d momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels using previous bar's data to avoid look-ahead
    # Upper band = highest high of previous 20 bars
    # Lower band = lowest low of previous 20 bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    highest_high_1d = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_1d - df_1d_close) / (highest_high_1d - lowest_low_1d) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r)
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Need enough bars for Donchian, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        williams_val = williams_r_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above Donchian upper band AND Williams %R oversold (< -80)
                if curr_high > highest_high[i] and williams_val < -80:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian lower band AND Williams %R overbought (> -20)
                elif curr_low < lowest_low[i] and williams_val > -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower band OR loss of volume confirmation
            if curr_low < lowest_low[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper band OR loss of volume confirmation
            if curr_high > highest_high[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dWilliamsR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0