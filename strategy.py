#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_v1
Hypothesis: Elder Ray (Bull/Bear Power) with 1d EMA13 trend filter and 6h volume confirmation.
Works in bull/bear: In uptrend (price > EMA13), buy when Bull Power > 0 and rising; in downtrend (price < EMA13), sell when Bear Power < 0 and falling.
Uses 6h timeframe for optimal trade frequency (target: 12-37 trades/year per symbol).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA13 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA13 on daily close
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    # Calculate Elder Ray components on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 of 6h close for Bull/Bear Power calculation
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13_6h  # Bull Power: High - EMA13
    bear_power = low - ema13_6h   # Bear Power: Low - EMA13
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema13_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.3 * vol_ma[i]
        
        # 1d trend filter: uptrend if price > EMA13, downtrend if price < EMA13
        uptrend = price > ema13_aligned[i]
        downtrend = price < ema13_aligned[i]
        
        if position == 0:
            # Long: uptrend + Bull Power > 0 and rising + volume
            if uptrend and bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + Bear Power < 0 and falling + volume
            elif downtrend and bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 or trend change to downtrend
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 or trend change to uptrend
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_v1"
timeframe = "6h"
leverage = 1.0