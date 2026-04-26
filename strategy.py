#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_1dTrend_v2
Hypothesis: 6h Elder Ray (Bull/Bear Power) zero-cross with 1d EMA trend filter.
- Long when 6h Bull Power crosses above zero AND price > 1d EMA34 (uptrend)
- Short when 6h Bear Power crosses below zero AND price < 1d EMA34 (downtrend)
- Uses 13-period EMA for Bull/Bear Power calculation (standard)
- Trend filter avoids counter-trend trades in strong markets
- Designed for low frequency (target 12-30 trades/year) with edge in both bull/bear regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray on 6h timeframe
    # Bull Power = High - EMA(close)
    # Bear Power = Low - EMA(close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate zero-cross signals
    bull_cross_above = (bull_power[1:] > 0) & (bull_power[:-1] <= 0)
    bear_cross_below = (bear_power[1:] < 0) & (bear_power[:-1] >= 0)
    
    # Pad to original length (shift due to diff)
    bull_cross_above = np.concatenate([[False], bull_cross_above])
    bear_cross_below = np.concatenate([[False], bear_cross_below])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 13 for 6h EMA)
    start_idx = max(34, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray zero-cross with 1d EMA trend filter
        if position == 0:
            # Long: Bull Power crosses above zero AND price > 1d EMA34 (uptrend)
            if bull_cross_above[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses below zero AND price < 1d EMA34 (downtrend)
            elif bear_cross_below[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power falls below zero
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power rises above zero
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_1dTrend_v2"
timeframe = "6h"
leverage = 1.0