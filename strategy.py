#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_v1
Hypothesis: Trade Elder Ray Bull/Bear Power on 6h with 1d EMA50 trend filter.
Elder Ray measures bull/bear strength relative to EMA13. Long when Bull Power > 0 and Bear Power < 0 with 1d uptrend.
Short when Bear Power < 0 and Bull Power > 0 with 1d downtrend. Works in both bull and bear markets by following the
1d trend while using 6h Elder Ray for precise entry/exit. Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13) and 1d EMA50 (50)
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: Bull Power > 0 (strong bulls) + Bear Power < 0 (weak bears) + 1d uptrend
            long_setup = (bull_power[i] > 0) and (bear_power[i] < 0) and htf_1d_bullish
            
            # Short setup: Bear Power < 0 (strong bears) + Bull Power > 0 (weak bulls) + 1d downtrend
            short_setup = (bear_power[i] < 0) and (bull_power[i] > 0) and htf_1d_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bull Power <= 0 (bulls weak) OR 1d trend turns bearish
            if (bull_power[i] <= 0) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power >= 0 (bears weak) OR 1d trend turns bullish
            if (bear_power[i] >= 0) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0