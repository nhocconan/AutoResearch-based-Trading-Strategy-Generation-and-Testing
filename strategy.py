#!/usr/bin/env python3
"""
4-hour Donchian(20) breakout with 12-hour EMA filter and volume confirmation
Hypothesis: Breakouts of Donchian(20) channels in the direction of the 12-hour EMA trend,
confirmed by volume > 1.5x 20-period average, capture momentum with fewer whipsaws.
Designed for ~20-30 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_ema_volume_v1"
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
    
    # 12-hour data for EMA filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12-hour EMA(20) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below EMA(20) or Donchian(10) low
            ema_close = ema_12h_aligned[i]
            donchian_low = np.min(low[max(0, i-10):i+1])
            if (close[i] < ema_close or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above EMA(20) or Donchian(10) high
            ema_close = ema_12h_aligned[i]
            donchian_high = np.max(high[max(0, i-10):i+1])
            if (close[i] > ema_close or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Donchian(20) channels
            donchian_high = np.max(high[max(0, i-20):i])
            donchian_low = np.min(low[max(0, i-20):i])
            
            # Long: price breaks above Donchian(20) high + volume spike + price > EMA
            if (close[i] > donchian_high and
                vol_spike[i] and
                close[i] > ema_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian(20) low + volume spike + price < EMA
            elif (close[i] < donchian_low and
                  vol_spike[i] and
                  close[i] < ema_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals