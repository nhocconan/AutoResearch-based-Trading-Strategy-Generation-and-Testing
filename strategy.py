#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_Volume
Hypothesis: Elder Ray (Bull/Bear Power) with 1d EMA20 trend filter and volume confirmation.
Bull Power = High - EMA13, Bear Power = EMA13 - Low.
Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) in uptrend (price > 1d EMA20).
Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) in downtrend (price < 1d EMA20).
Works in bull/bear by following 1d trend and using momentum exhaustion signals.
Target: 15-35 trades/year on 6h to avoid fee drag.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 (13) and enough for trend
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish momentum (Bull Power > 0 AND Bear Power < 0) AND uptrend (price > EMA20) AND volume
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_20_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum (Bear Power > 0 AND Bull Power < 0) AND downtrend (price < EMA20) AND volume
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_20_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum turns bearish OR trend turns bearish
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum turns bullish OR trend turns bullish
            if bear_power[i] <= 0 or bull_power[i] >= 0 or close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals