#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum. 
Using 1d EMA50 as trend filter ensures alignment with higher timeframe direction. 
Volume confirmation (current volume > 2.0 * 20-bar average) filters weak breakouts. 
Designed for 12h timeframe with 50-150 total trades over 4 years to minimize fee drag. 
Works in bull markets (trend continuation) and bear markets (trend continuation down).
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
    
    # Get daily data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need 50 for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 12h timeframe
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA50, and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above upper Donchian channel with volume confirmation in uptrend
            long_breakout = (curr_close > upper_channel) and volume_confirm and uptrend
            # Short: price breaks below lower Donchian channel with volume confirmation in downtrend
            short_breakout = (curr_close < lower_channel) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below lower Donchian channel OR EMA50 trend turns down
            if curr_close < lower_channel or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian channel OR EMA50 trend turns up
            if curr_close > upper_channel or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0