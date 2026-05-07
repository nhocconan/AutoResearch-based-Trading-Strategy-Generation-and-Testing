# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Elder_Ray_Cross_1wTrend_v1
Hypothesis: Use Elder Ray (Bull/Bear Power) crossovers on 6h with weekly trend filter to capture momentum shifts.
- Bull Power = High - EMA13, Bear Power = Low - EMA13
- Long when Bull Power crosses above zero with weekly uptrend
- Short when Bear Power crosses below zero with weekly downtrend
- Weekly EMA40 trend filter ensures alignment with higher timeframe
- Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag
- Works in bull/bear via trend filter: only trade in direction of weekly trend
"""
name = "6h_Elder_Ray_Cross_1wTrend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    ema_40_weekly = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_weekly_aligned = align_htf_to_ltf(prices, df_1w, ema_40_weekly)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Crossovers: Bull Power crosses above zero, Bear Power crosses below zero
    bull_cross_up = (bull_power[1:] > 0) & (bull_power[:-1] <= 0)
    bear_cross_down = (bear_power[1:] < 0) & (bear_power[:-1] >= 0)
    
    # Pad to original length
    bull_cross_up = np.concatenate([[False], bull_cross_up])
    bear_cross_down = np.concatenate([[False], bear_cross_down])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 40)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema_40_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero + weekly uptrend
            if bull_cross_up[i] and close[i] > ema_40_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses below zero + weekly downtrend
            elif bear_cross_down[i] and close[i] < ema_40_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power crosses below zero
            if bear_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power crosses above zero
            if bull_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals