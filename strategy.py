#!/usr/bin/env python3
"""
#100843 - 4h_Aroon_Oscillator_1dTrend_12hVolumeFilter
Hypothesis: Aroon oscillator identifies trend strength with low lag. Combined with 1d trend filter and 12h volume confirmation,
it captures strong trends while avoiding chop. Works in bull (strong uptrends) and bear (strong downtrends).
Target: 20-40 trades/year to minimize fee drag. Uses 4h primary with 1d HTF for trend and 12h for volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def aroon_oscillator(high, low, period=25):
    """Aroon Oscillator = Aroon Up - Aroon Down, ranges -100 to +100"""
    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period, n):
        # Periods since highest high
        high_idx = i - np.argmax(high[i-period+1:i+1])
        aroon_up[i] = ((period - high_idx) / period) * 100
        
        # Periods since lowest low
        low_idx = i - np.argmin(low[i-period+1:i+1])
        aroon_down[i] = ((period - low_idx) / period) * 100
    
    return aroon_up - aroon_down

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume MA20
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Aroon oscillator (25 period) on 4h
    aroon = aroon_oscillator(high, low, 25)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(aroon[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: Aroon > 50 (uptrend), above 1d EMA50, volume > 1.5x 12h MA
        if (aroon[i] > 50 and 
            close[i] > ema50_1d_aligned[i] and 
            volume[i] > (vol_ma_12h_aligned[i] * 1.5)):
            signals[i] = 0.25
            position = 1
        # Short condition: Aroon < -50 (downtrend), below 1d EMA50, volume > 1.5x 12h MA
        elif (aroon[i] < -50 and 
              close[i] < ema50_1d_aligned[i] and 
              volume[i] > (vol_ma_12h_aligned[i] * 1.5)):
            signals[i] = -0.25
            position = -1
        # Exit conditions: Aroon crosses zero (trend change)
        elif position == 1 and aroon[i] < 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and aroon[i] > 0:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Aroon_Oscillator_1dTrend_12hVolumeFilter"
timeframe = "4h"
leverage = 1.0