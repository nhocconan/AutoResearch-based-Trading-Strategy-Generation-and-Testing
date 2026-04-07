#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout with 1d EMA Trend and Volume Confirmation
# Hypothesis: Breakouts above/below 12h Donchian(20) channels in direction of 1d EMA(50) trend
# capture strong momentum moves, with volume filter ensuring institutional participation.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d EMA to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 12h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_avg[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower or trend changes
            if close[i] < lowest_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper or trend changes
            if close[i] > highest_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: breakout above upper channel in uptrend
            if close[i] > highest_high[i] and close[i] > ema_50_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: breakout below lower channel in downtrend
            elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals