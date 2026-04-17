#!/usr/bin/env python3
"""
1h_VolumeSpike_PullbackToEMA_v1
Long when: 1) 4h EMA20 uptrend, 2) 1d volume > 1.5x 20-period avg, 3) 1h close > EMA20 after pullback.
Short when: 1) 4h EMA20 downtrend, 2) 1d volume > 1.5x 20-period avg, 3) 1h close < EMA20 after pullback.
Exit when price crosses EMA20 in opposite direction.
Volume spike filters for institutional interest; EMA20 provides dynamic support/resistance.
Target: 60-150 total trades over 4 years (15-37/year).
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
    
    # === 1h EMA20 (dynamic support/resistance) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 4h EMA20 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === 1d volume average (20-period) for spike detection ===
    df_1d = get_htf_data(prices, '1d')
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20[i]) or 
            np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(volume[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike condition: current 1h volume > 1.5x 1d average volume
        volume_spike = volume[i] > 1.5 * vol_avg_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: 4h uptrend, volume spike, pullback to EMA20 then bounce
            if (ema20_4h_aligned[i] > ema20_4h_aligned[i-1] and  # 4h EMA20 rising
                volume_spike and
                close[i] > ema20[i] and  # price above EMA20
                close[i-1] <= ema20[i-1]):  # was at or below EMA20 previous bar
                signals[i] = 0.20
                position = 1
                continue
            # Short: 4h downtrend, volume spike, pullback to EMA20 then reject
            elif (ema20_4h_aligned[i] < ema20_4h_aligned[i-1] and  # 4h EMA20 falling
                  volume_spike and
                  close[i] < ema20[i] and  # price below EMA20
                  close[i-1] >= ema20[i-1]):  # was at or above EMA20 previous bar
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: price crosses EMA20 in opposite direction
        elif position == 1:
            # Exit long: price crosses below EMA20
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_PullbackToEMA_v1"
timeframe = "1h"
leverage = 1.0