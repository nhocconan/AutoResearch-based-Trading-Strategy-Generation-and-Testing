#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
- Camarilla levels: R4 = close + 1.1*(high-low)/2, S4 = close - 1.1*(high-low)/2 (wider breakout levels than R3/S3 for even fewer false signals)
- Long: price breaks above R4 + price > 1d EMA50 (uptrend) + volume > 2.0x 24-period avg
- Short: price breaks below S4 + price < 1d EMA50 (downtrend) + volume > 2.0x 24-period avg
- Exit: price crosses 1d EMA50 (trend-based exit)
- Uses even wider Camarilla levels (R4/S4) to further reduce whipsaws and false breakouts vs R3/S3
- Volume confirmation ensures breakout validity
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 2.0x 24-period average (strict spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R4, S4)
    # R4 = close + 1.1*(high-low)/2, S4 = close - 1.1*(high-low)/2
    camarilla_r4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50)  # Need 24 for volume MA, 50 for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R4 + price > 1d EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > camarilla_r4_aligned[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + price < 1d EMA50 (downtrend) + volume spike
            elif volume_spike and close[i] < camarilla_s4_aligned[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA50 (trend-based exit)
            if close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA50 (trend-based exit)
            if close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0