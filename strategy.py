#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme with 1d EMA34 trend filter and volume spike.
- Primary timeframe: 4h, HTF: 1d for trend filter
- Williams %R(14): extreme oversold (< -80) for long, extreme overbought (> -20) for short
- Trend filter: price > 1d EMA34 for long bias, price < 1d EMA34 for short bias
- Volume confirmation: volume > 2.0x 20-period average to avoid low-liquidity false signals
- Exit: Williams %R returns to neutral zone (-50) or opposite extreme triggers reversal
- Discrete position sizing: ±0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Works in bull markets (buy extreme dips in uptrend) and bear markets (sell extreme rallies in downtrend)
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
    
    # Williams %R(14) calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 2.0x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R extreme oversold (< -80) + price > 1d EMA34 (uptrend) + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extreme overbought (> -20) + price < 1d EMA34 (downtrend) + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to neutral zone (-50) or extreme overbought triggers reversal
            if williams_r[i] >= -50 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to neutral zone (-50) or extreme oversold triggers reversal
            if williams_r[i] <= -50 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0