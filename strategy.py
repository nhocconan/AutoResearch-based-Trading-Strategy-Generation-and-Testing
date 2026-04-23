#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
- Williams Alligator (jaw=13, teeth=8, lips=5) defines market structure: 
  Alligator sleeping (jaw>teeth>lips or jaw<teeth<lips) = trend, intertwined = ranging
- Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13 measures trend strength
- Enter long when: Alligator awake (trending) AND Bull Power > 0 AND price > EMA13 AND volume > 1.5x average
- Enter short when: Alligator awake (trending) AND Bear Power < 0 AND price < EMA13 AND volume > 1.5x average
- Use 1d EMA50 as trend filter: only long above, short below to avoid counter-trend whipsaw
- Position size: 0.25 discrete level to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Works in both bull/bear via 1d trend filter + volatility-adjusted Alligator signals
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
    
    # Williams Alligator components (Smoothed Moving Average = SMA with specific periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray components (using EMA13 for consistency)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 50)  # Alligator lips, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator: check if sleeping (intertwined) or awake (separated)
        # Alligator sleeping when jaws, teeth, lips are intertwined (no clear trend)
        alligator_sleeping = (
            (jaw[i] >= teeth[i] >= lips[i]) or  # All descending order
            (jaw[i] <= teeth[i] <= lips[i])     # All ascending order
        )
        alligator_awake = not alligator_sleeping  # Jaws, teeth, lips separated = trending
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Alligator awake (trending) AND Bull Power > 0 AND price > EMA13 AND price > 1d EMA50 AND volume confirmation
            if (alligator_awake and 
                bull_power[i] > 0 and 
                close[i] > ema13[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Alligator awake (trending) AND Bear Power < 0 AND price < EMA13 AND price < 1d EMA50 AND volume confirmation
            elif (alligator_awake and 
                  bear_power[i] < 0 and 
                  close[i] < ema13[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator sleeping (loss of trend) OR Bear Power < 0 OR price crosses below EMA13 OR price crosses below 1d EMA50
            if (not alligator_awake or 
                bear_power[i] < 0 or 
                close[i] < ema13[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator sleeping (loss of trend) OR Bull Power > 0 OR price crosses above EMA13 OR price crosses above 1d EMA50
            if (not alligator_awake or 
                bull_power[i] > 0 or 
                close[i] > ema13[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0