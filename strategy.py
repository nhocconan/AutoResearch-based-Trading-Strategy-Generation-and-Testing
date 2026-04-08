#!/usr/bin/env python3
"""
4h Donchian(20) breakout with 12h trend filter and volume confirmation
Hypothesis: Price breaking Donchian channels in direction of 12h EMA(50) trend with volume spike (current > 2x 20-period avg) captures momentum. 12h trend filter reduces whipsaws, volume confirms institutional interest. Target: 30-60 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR price breaks below Donchian(10) low
            donchian_low = np.min(low[max(0, i-10):i+1])
            if (close[i] <= ema_50_12h_aligned[i] or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price breaks above Donchian(10) high
            donchian_high = np.max(high[max(0, i-10):i+1])
            if (close[i] >= ema_50_12h_aligned[i] or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Donchian(20) channels
            donchian_high = np.max(high[max(0, i-20):i])
            donchian_low = np.min(low[max(0, i-20):i])
            
            # Long: price breaks above Donchian(20) high + volume spike + uptrend
            if (close[i] > donchian_high and
                close[i] > ema_50_12h_aligned[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian(20) low + volume spike + downtrend
            elif (close[i] < donchian_low and
                  close[i] < ema_50_12h_aligned[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals