#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Donchian(20) breakouts on 12h with 1d EMA50 trend filter and volume spikes capture strong trends while avoiding whipsaws. 
Uses discrete sizing (0.25) to limit drawdown. Designed for low trade frequency (target 20-50/year) to minimize fee drag. 
Works in bull markets via breakout momentum and in bear markets via trend filter preventing counter-trend trades.
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
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) channels on 12h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA50 (50), Donchian (20), volume avg (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Long when price breaks above Donchian upper with volume and uptrend
            if close_val > upper and vol_conf and close_val > ema50:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short when price breaks below Donchian lower with volume and downtrend
            elif close_val < lower and vol_conf and close_val < ema50:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: price re-enters Donchian channel or trend reverses
            if close_val < upper and close_val > lower:
                signals[i] = 0.0
                position = 0
            elif close_val < ema50:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters Donchian channel or trend reverses
            if close_val < upper and close_val > lower:
                signals[i] = 0.0
                position = 0
            elif close_val > ema50:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0