#!/usr/bin/env python3
"""
6h_Stochastic_Breakout_1dTrend_Volume
Hypothesis: Trade breakouts at 6-hour Donchian channels with 1-day Stochastic trend filter and volume confirmation.
This strategy targets 15-35 trades per year per symbol (60-140 total over 4 years) by using tight entry conditions:
- Breakout above/below 6-hour Donchian(20) channel
- Confirmed by 1-day Stochastic > 60 (uptrend) or < 40 (downtrend)
- Volume spike > 1.5x 20-period EMA on 6h
Works in bull/bear markets by aligning with the daily momentum direction via Stochastic.
"""

name = "6h_Stochastic_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Stochastic for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Stochastic %K (14,3,3)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    stoch_k = 100 * (df_1d['close'].values - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_k = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values
    stoch_k = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values  # %D
    
    # Align Stochastic to 6h timeframe
    stoch_6h = align_htf_to_ltf(prices, df_1d, stoch_k)
    
    # === 6h Donchian Channel (20) ===
    highest_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Filter (1.5x 20-period EMA on 6h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and Stochastic calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or 
            np.isnan(stoch_6h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with bullish Stochastic and volume
            if (close[i] > highest_high_6h[i] and 
                stoch_6h[i] > 60 and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below Donchian low with bearish Stochastic and volume
            elif (close[i] < lowest_low_6h[i] and 
                  stoch_6h[i] < 40 and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low (reversal)
            if close[i] < lowest_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above Donchian high (reversal)
            if close[i] > highest_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals