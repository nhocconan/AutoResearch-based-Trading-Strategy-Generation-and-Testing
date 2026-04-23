#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h, HTF: 1d for trend filter
- Williams %R(14): long when < -80 (oversold), short when > -20 (overbought)
- Trend filter: price > 1d EMA34 for longs, price < 1d EMA34 for shorts
- Volume confirmation: volume > 1.5x 20-period average to avoid false signals
- Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R from 6h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Volume MA, Williams %R, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + price > 1d EMA34 (uptrend) + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 1d EMA34 (downtrend) + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back above -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back below -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0