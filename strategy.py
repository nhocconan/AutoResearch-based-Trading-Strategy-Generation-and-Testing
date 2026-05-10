#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_Volume
Hypothesis: Elder Ray index (Bull Power = High - EMA, Bear Power = EMA - Low) with 13-period EMA.
Use 1d EMA34 as trend filter. Enter long when Bull Power > 0 and rising in uptrend, short when Bear Power < 0 and falling in downtrend.
Add volume confirmation (current volume > 1.5x 20 EMA volume). Target: 15-25 trades/year.
Works in bull/bear via trend filter and momentum confirmation.
"""

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components (13-period EMA)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    
    # Volume filter: current volume > 1.5x 20-period EMA (moderate to balance frequency)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 and 1d EMA34
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend: price vs EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: uptrend AND Bull Power > 0 AND Bull Power rising (current > previous) with volume
            if (uptrend and bull_power[i] > 0 and 
                i > 0 and bull_power[i] > bull_power[i-1] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND Bear Power > 0 AND Bear Power rising (current > previous) with volume
            elif (downtrend and bear_power[i] > 0 and 
                  i > 0 and bear_power[i] > bear_power[i-1] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR trend changes to downtrend
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 OR trend changes to uptrend
            if bear_power[i] <= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals