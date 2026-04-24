#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trendless markets when lines are intertwined.
- Trades only when Alligator is "awake" (lines separated) and aligned with 1d EMA50 trend.
- Volume confirmation (>1.3x 20-bar average) filters low-conviction signals.
- Designed for 6h timeframe to capture medium-term swings in both bull and bear markets.
- Target trades: 90-180 total over 4 years (22-45/year) to balance opportunity and fee drag.
"""

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
    
    # Get 1d data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h close: Jaw(13), Teeth(8), Lips(5)
    # Smoothed with 3-period offset as per original formula
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13) + 8  # Need enough for EMA and Alligator (max shift 8)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms and Alligator is awake (not intertwined)
            if volume_confirm:
                # Alligator awake: Jaw, Teeth, Lips are separated (not intertwined)
                # Long condition: Lips > Teeth > Jaw AND above 1d EMA50
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short condition: Lips < Teeth < Jaw AND below 1d EMA50
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator starts to sleep (lines intertwine) OR crosses below 1d EMA50
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator starts to sleep (lines intertwine) OR crosses above 1d EMA50
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0