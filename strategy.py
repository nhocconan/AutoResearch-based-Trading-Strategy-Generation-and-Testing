#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_v1
Hypothesis: Trade 6h Elder Ray Bull/Bear Power extremes with 1d EMA34 trend filter.
In bull markets: buy when Bear Power shows exhaustion (less negative) and price above 1d EMA.
In bear markets: sell when Bull Power shows exhaustion (less positive) and price below 1d EMA.
Uses 1d EMA34 for trend filter and Elder Ray for mean-reversion entries at extremes.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Uses discrete position sizing (0.0, ±0.25) to reduce fee churn.
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d  # Using 6h high minus 1d EMA13
    bear_power = low - ema_13_1d   # Using 6h low minus 1d EMA13
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA periods
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Bull Power weakening (less negative) AND 1d uptrend
            # Bull Power = High - EMA13, so less negative means closer to zero or positive
            long_signal = (bull_power_aligned[i] > -0.1 * close_val) and trend_1d_up
            
            # Short: Bear Power weakening (less positive) AND 1d downtrend
            # Bear Power = Low - EMA13, so less positive means closer to zero or negative
            short_signal = (bear_power_aligned[i] < 0.1 * close_val) and trend_1d_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power turns negative again OR trend flips down
            if (bull_power_aligned[i] < 0) or (not trend_1d_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power turns positive again OR trend flips up
            if (bear_power_aligned[i] > 0) or (not trend_1d_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_v1"
timeframe = "6h"
leverage = 1.0