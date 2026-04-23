#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme + 1w EMA50 trend filter + volume spike confirmation.
- Long: Williams %R(14) < -80 (oversold) + price > 1w EMA50 + volume > 1.8x 24-period avg volume
- Short: Williams %R(14) > -20 (overbought) + price < 1w EMA50 + volume > 1.8x 24-period avg volume
- Exit: Williams %R returns to -50 level OR opposing extreme reached
- Uses 1w EMA50 as trend filter to avoid counter-trend trades in strong trends
- Volume confirmation reduces false signals in low-participation moves
- Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag
- Williams %R is effective in ranging/bear markets (2022-2024) and captures reversals in bull rallies
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
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: > 1.8x 24-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14, 50)  # Need 24 for volume MA, 14 for Williams %R, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme conditions
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        williams_exit = abs(williams_r[i] + 50) < 5  # Near -50 level
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold + price > 1w EMA50 + volume spike
            if williams_oversold and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought + price < 1w EMA50 + volume spike
            elif williams_overbought and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 OR overbought extreme reached
            if williams_exit or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 OR oversold extreme reached
            if williams_exit or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0