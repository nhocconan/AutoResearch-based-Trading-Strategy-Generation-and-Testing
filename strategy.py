#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h, HTF: 4h for trend filter
- Long: Close breaks above R1 + price > 4h EMA50 (uptrend) + volume > 1.5x 20-period avg
- Short: Close breaks below S1 + price < 4h EMA50 (downtrend) + volume > 1.5x 20-period avg
- Exit: Close reverts to pivot point (PP) of Camarilla levels
- Session filter: 08-20 UTC to reduce noise trades
- Position size: 0.20 (discrete levels to minimize fee churn)
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Uses mtf_data helper for proper HTF alignment without look-ahead
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
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day using 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    range_1d = high_1d - low_1d
    r1 = close_1d + 1.1 * range_1d / 12.0
    s1 = close_1d - 1.1 * range_1d / 12.0
    pp = (high_1d + low_1d + close_1d) / 3.0  # Pivot point
    
    # Align to 1h timeframe (values from previous 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R1 + price > 4h EMA50 (uptrend) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S1 + price < 4h EMA50 (downtrend) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close reverts to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close reverts to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0