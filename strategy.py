#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter.
Long when price breaks above upper Donchian channel (20-period high) AND 1d volume > 1.5x 20-bar average volume AND chop > 61.8 (range regime).
Short when price breaks below lower Donchian channel (20-period low) AND 1d volume > 1.5x 20-bar average volume AND chop > 61.8.
Exit when price touches the opposite Donchian channel level or after 5 bars (time-based exit).
Uses 1d for volume confirmation and chop regime, 12h for execution and Donchian channels.
Designed to capture breakouts in ranging markets with volume confirmation. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time'].values  # for tracking bars in trade
    
    # Get 1d data for volume and chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA for confirmation (20-bar)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d chop regime (choppiness index)
    # True range
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    log_sum = np.log10(sum_atr14 + 1e-10)
    log_n = np.log10(14)
    chop = 100 * log_sum / log_n
    
    # Calculate 12h Donchian channels (20-period)
    # Need to resample to 12h for Donchian calculation, but we'll use HTF helper
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel: 20-period high
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align all 1d indicators to 12h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align 12h Donchian channels to 12h timeframe (no alignment needed as we're already on 12h)
    # But we need to align to the prices timeframe (which is 12h per experiment instructions)
    # Since prices is already 12h timeframe, we can use the values directly
    upper_20_aligned = upper_20
    lower_20_aligned = lower_20
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    bars_in_trade = 0  # track bars in current trade for time-based exit
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            bars_in_trade = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-bar average
        # We need to get the 1d volume for the current 12h bar
        # Since we're on 12h timeframe, we use the aligned 1d volume MA
        volume_confirmed = volume_1d[i // 12] > 1.5 * vol_ma_20_1d_aligned[i] if i // 12 < len(volume_1d) else False
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_upper = close[i] > upper_20_aligned[i]
        breakout_lower = close[i] < lower_20_aligned[i]
        
        # Exit conditions: touch opposite channel or time-based exit (5 bars)
        touch_opposite = (position == 1 and close[i] < lower_20_aligned[i]) or \
                         (position == -1 and close[i] > upper_20_aligned[i])
        time_exit = bars_in_trade >= 5
        
        if position == 0:
            # Long: break above upper channel with volume confirmation and chop regime
            if (breakout_upper and volume_confirmed and chop_filter):
                signals[i] = 0.25
                position = 1
                bars_in_trade = 1
            # Short: break below lower channel with volume confirmation and chop regime
            elif (breakout_lower and volume_confirmed and chop_filter):
                signals[i] = -0.25
                position = -1
                bars_in_trade = 1
        
        elif position == 1:
            # Exit long: touch lower channel or time-based exit
            if (touch_opposite or time_exit):
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
                bars_in_trade += 1
        
        elif position == -1:
            # Exit short: touch upper channel or time-based exit
            if (touch_opposite or time_exit):
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
                bars_in_trade += 1
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0