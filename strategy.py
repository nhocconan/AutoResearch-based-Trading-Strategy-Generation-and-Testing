#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d EMA Trend and Volume Spike
Hypothesis: Breakouts above/below Donchian channels capture momentum in the direction of the higher timeframe trend.
1d EMA50 filters trend direction, volume spikes confirm institutional participation.
Designed to work in both bull and bear markets by following the trend.
Target: 25-40 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
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
    
    # Donchian Channel (20-period) - breakout signals
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    high_roll = high_series.rolling(window=20, min_periods=20).max().values
    low_roll = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume Spike Detector - 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # 1d EMA50 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data is not ready
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d EMA50 (trend change)
            if close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d EMA50 (trend change)
            if close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high + price above 1d EMA50 + volume spike
            if (close[i] > high_roll[i-1] and 
                close[i] > ema_50_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low + price below 1d EMA50 + volume spike
            elif (close[i] < low_roll[i-1] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals