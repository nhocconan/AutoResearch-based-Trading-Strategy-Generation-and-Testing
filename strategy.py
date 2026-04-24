#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + 1w EMA50 trend filter with volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 1d EMA13)
- Long when Bull Power > 0 and increasing, Bear Power < 0 and decreasing (bullish momentum)
- Short when Bear Power > 0 and increasing, Bull Power < 0 and decreasing (bearish momentum)
- 1w EMA50 ensures alignment with weekly trend (avoid counter-trend whipsaws)
- Volume spike (>1.8x 20-period average) confirms conviction
- Works in bull/bear via trend filter and momentum-based entries
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
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
    
    # Get 1d data ONCE before loop for EMA13 (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1d EMA13 to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray Power (using aligned 1d EMA13)
    bull_power = high - ema_13_1d_aligned  # High - EMA13
    bear_power = ema_13_1d_aligned - low   # EMA13 - Low
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate power momentum (change from previous bar)
        if i > 0:
            bull_power_momentum = bull_power[i] - bull_power[i-1]
            bear_power_momentum = bear_power[i] - bear_power[i-1]
        else:
            bull_power_momentum = 0
            bear_power_momentum = 0
        
        if position == 0:
            # Long: Bull Power > 0 and increasing, Bear Power < 0 and decreasing
            #         with volume spike and price above weekly EMA50 (bullish alignment)
            if (bull_power[i] > 0 and bull_power_momentum > 0 and 
                bear_power[i] < 0 and bear_power_momentum < 0 and
                volume_spike[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 and increasing, Bull Power < 0 and decreasing
            #        with volume spike and price below weekly EMA50 (bearish alignment)
            elif (bear_power[i] > 0 and bear_power_momentum > 0 and
                  bull_power[i] < 0 and bull_power_momentum < 0 and
                  volume_spike[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR Bear Power turns positive
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative OR Bull Power turns positive
            if bear_power[i] <= 0 or bull_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0