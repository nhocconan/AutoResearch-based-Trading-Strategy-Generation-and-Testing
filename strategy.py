#!/usr/bin/env python3
"""
12H_Donchian20_VolumeSpike_1dTrendFilter
Hypothesis: Donchian(20) breakouts on 12h with volume confirmation and 1d EMA50 trend filter capture sustained moves in both bull and bear markets.
The 12h timeframe reduces trade frequency to avoid fee drag, while volume and trend filters ensure high-probability entries.
Designed for ~15-25 trades/year to minimize fee impact in 2025 ranging markets.
"""

name = "12H_Donchian20_VolumeSpike_1dTrendFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above prior 20-period high
        breakout_down = close[i] < lowest_low[i-1]   # Break below prior 20-period low
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Upward breakout + uptrend + volume spike
            if breakout_up and uptrend and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Downward breakout + downtrend + volume spike
            elif breakout_down and downtrend and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Reverse breakout or trend change
            if position == 1:
                # Exit: Downward breakout or loss of uptrend
                if breakout_down or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Upward breakout or loss of downtrend
                if breakout_up or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals