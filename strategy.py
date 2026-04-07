#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_volume_filter_v1
Hypothesis: 12h Donchian breakout (20-period) with volume confirmation and 1d EMA50 trend filter.
Enters long when price breaks above upper Donchian channel with volume > 20-period average and price above 1d EMA50.
Enters short when price breaks below lower Donchian channel with volume > 20-period average and price below 1d EMA50.
Uses Donchian breakout for clear breakout signals, volume filter ensures momentum confirmation,
and 1d trend filter prevents counter-trend trades. Designed for 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel: 20-period high/low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > high_roll[i-1]
        breakout_down = close[i] < low_roll[i-1]
        
        # 1d trend filter
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or below 1d EMA50
            if close[i] < low_roll[i-1] or below_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or above 1d EMA50
            if close[i] > high_roll[i-1] or above_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout up with volume confirmation and above 1d EMA50
            if breakout_up and vol_confirmed and above_1d_ema50:
                position = 1
                signals[i] = 0.25
            # Short: breakout down with volume confirmation and below 1d EMA50
            elif breakout_down and vol_confirmed and below_1d_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals