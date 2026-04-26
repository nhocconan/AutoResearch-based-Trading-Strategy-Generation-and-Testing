#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_12hTrend_VolumeConfirm_v2
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 12h EMA34 trend filter and volume confirmation. 
In bull markets (price > 12h EMA34), look for Bull Power expansion with volume to go long. 
In bear markets (price < 12h EMA34), look for Bear Power expansion with volume to go short. 
Volume confirmation ensures breakouts have conviction. Targeting 80-150 total trades over 4 years (20-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 34, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend filter
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Long logic: Bull Power expansion with volume confirmation in uptrend
        if (bull_power[i] > 0 and 
            bull_power[i] > bull_power[i-1] and  # Expanding bull power
            volume_confirm[i] and 
            uptrend):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Bear Power expansion with volume confirmation in downtrend
        elif (bear_power[i] > 0 and 
              bear_power[i] > bear_power[i-1] and  # Expanding bear power
              volume_confirm[i] and 
              downtrend):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: power contraction or trend reversal
        elif position == 1 and (bull_power[i] <= 0 or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bear_power[i] <= 0 or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hTrend_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0