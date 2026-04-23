#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1w EMA50 Trend Filter + Volume Spike (>2.0x average)
- Williams %R(14) identifies overbought/oversold extremes (long when %R < -80, short when %R > -20)
- 1w EMA50 ensures trading with the weekly trend to avoid counter-trend whipsaws
- Volume confirmation (>2.0x 24-bar average) filters low-conviction breakouts
- Uses 6h timeframe to target 12-37 trades/year (50-150 over 4 years) minimizing fee drag
- Designed to work in both bull and bear markets via trend filter and mean-reversion extremes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 24)  # EMA50, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R extreme signals
        oversold = williams_r[i] < -80  # Extreme oversold
        overbought = williams_r[i] > -20  # Extreme overbought
        
        if position == 0:
            # Long: Williams %R oversold AND price > 1w EMA50 AND volume confirmation
            if oversold and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price < 1w EMA50 AND volume confirmation
            elif overbought and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R rises above -50 (momentum fading) OR price < 1w EMA50 (trend flip)
            if williams_r[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R falls below -50 (momentum fading) OR price > 1w EMA50 (trend flip)
            if williams_r[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0