#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
# Long when price breaks above 20-period high + EMA50 up + volume > 1.5x average.
# Short when price breaks below 20-period low + EMA50 down + volume > 1.5x average.
# Exit on opposite Donchian breakout or trend reversal.
# Target: 80-180 total trades over 4 years (20-45/year).
name = "4h_Donchian20_1dEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d timeframe
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)  # Previous day's EMA for trend
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_up = ema50_1d > ema50_1d_prev
    ema50_down = ema50_1d < ema50_1d_prev
    
    # Align EMA50 trend to 4h timeframe
    ema50_up_aligned = align_htf_to_ltf(prices, df_1d, ema50_up)
    ema50_down_aligned = align_htf_to_ltf(prices, df_1d, ema50_down)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema50_up_aligned[i]) or np.isnan(ema50_down_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above Donchian high + EMA50 up + volume
            if close[i] > high_max[i] and ema50_up_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low + EMA50 down + volume
            elif close[i] < low_min[i] and ema50_down_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian low break or EMA50 turns down
            if close[i] < low_min[i] or not ema50_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian high break or EMA50 turns up
            if close[i] > high_max[i] or not ema50_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals