#!/usr/bin/env python3
"""
4-hour Donchian(30) breakout with 1-day trend filter and volume confirmation
Hypothesis: Breakouts of Donchian(30) channels in the direction of the 1-day EMA(50) trend,
confirmed by volume > 1.8x 20-period average, capture institutional momentum. 
The 1-day trend filter avoids counter-trend whipsaws, while volume confirms strength.
Designed for fewer trades (<50/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
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
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-day EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR price breaks below Donchian(15) low
            donchian_low = np.min(low[max(0, i-15):i+1])
            if (close[i] <= ema_50_1d_aligned[i] or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price breaks above Donchian(15) high
            donchian_high = np.max(high[max(0, i-15):i+1])
            if (close[i] >= ema_50_1d_aligned[i] or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Donchian(30) channels - wider for fewer, higher-quality signals
            donchian_high = np.max(high[max(0, i-30):i])
            donchian_low = np.min(low[max(0, i-30):i])
            
            # Long: price breaks above Donchian(30) high + volume spike + uptrend
            if (close[i] > donchian_high and
                close[i] > ema_50_1d_aligned[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian(30) low + volume spike + downtrend
            elif (close[i] < donchian_low and
                  close[i] < ema_50_1d_aligned[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals