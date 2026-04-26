#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_12hTrend_VolumeConfirm_v3
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h with 12h EMA34 trend filter and volume confirmation. 
Long when Bull Power > 0 AND price > EMA34_12h AND volume spike. 
Short when Bear Power < 0 AND price < EMA34_12h AND volume spike. 
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. 
Designed to work in both bull (trend following) and bear (counter-trend via EMA filter) markets.
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5 * 20-period EMA of volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(ema13[i]) or
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
        
        # Long logic: Bull Power > 0 AND price > EMA34_12h AND volume spike
        if bull_power[i] > 0 and close[i] > ema34_12h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: Bear Power < 0 AND price < EMA34_12h AND volume spike
        elif bear_power[i] < 0 and close[i] < ema34_12h_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite power signal or loss of volume confirmation
        elif position == 1 and (bull_power[i] <= 0 or close[i] <= ema34_12h_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bear_power[i] >= 0 or close[i] >= ema34_12h_aligned[i]):
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

name = "6h_ElderRay_BullBearPower_12hTrend_VolumeConfirm_v3"
timeframe = "6h"
leverage = 1.0