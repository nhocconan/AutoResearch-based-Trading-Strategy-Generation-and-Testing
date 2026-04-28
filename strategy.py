#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_Filter
Hypothesis: Uses Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) on 6h timeframe.
Trades only in direction of weekly EMA20 trend to avoid counter-trend whipsaws.
Requires both Bull/Bear Power expansion (increasing momentum) and weekly trend alignment.
Designed for low trade frequency (12-37/year) to minimize fee drift while capturing strong momentum moves.
Works in bull markets via long signals and bear markets via short signals.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Smooth the power signals (2-period EMA) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # Momentum: increasing power (current > previous)
    bull_power_increasing = bull_power_smooth > np.roll(bull_power_smooth, 1)
    bear_power_increasing = bear_power_smooth > np.roll(bear_power_smooth, 1)
    # Handle first element
    bull_power_increasing[0] = False
    bear_power_increasing[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(ema13[i]) or
            np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: power increasing in trend direction
        long_entry = bull_power_increasing[i] and uptrend
        short_entry = bear_power_increasing[i] and downtrend
        
        # Exit conditions: power decreasing OR trend reversal
        long_exit = not bull_power_increasing[i] or not uptrend
        short_exit = not bear_power_increasing[i] or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0