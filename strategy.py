#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
Long when price breaks above 20-bar high AND 6h close > 1d EMA50 AND 6h volume > 1.5x 20-bar average.
Short when price breaks below 20-bar low AND 6h close < 1d EMA50 AND 6h volume > 1.5x 20-bar average.
Exit when price touches the opposite Donchian level (20-bar low for long, 20-bar high for short).
Uses 1d for EMA50 trend filter, 6h for Donchian channels and volume confirmation.
Designed to capture medium-term trends with volume confirmation, working in both bull and bear markets.
Target: 12-30 trades/year per symbol (50-120 total over 4 years).
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
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        above_ema50 = close[i] > ema50_1d_aligned[i]
        below_ema50 = close[i] < ema50_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_high = close[i] > high_ma_20[i]
        breakout_low = close[i] < low_ma_20[i]
        
        # Exit conditions: touch opposite Donchian level
        exit_long = close[i] < low_ma_20[i]  # touch 20-bar low
        exit_short = close[i] > high_ma_20[i]  # touch 20-bar high
        
        if position == 0:
            # Long: break above 20-bar high with volume confirmation and above 1d EMA50
            if (breakout_high and volume_confirmed and above_ema50):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-bar low with volume confirmation and below 1d EMA50
            elif (breakout_low and volume_confirmed and below_ema50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch 20-bar low
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch 20-bar high
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_Volume_Trend"
timeframe = "6h"
leverage = 1.0