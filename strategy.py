#!/usr/bin/env python3
# 6H_1D_ElderRay_BullBearPower_Trend_Follow
# Hypothesis: Use daily Elder Ray (bull/bear power) with 13-day EMA for trend direction.
# Bull power = High - EMA13, Bear power = EMA13 - Low.
# Long when bull power > 0 and rising + price above EMA13.
# Short when bear power > 0 and rising + price below EMA13.
# Works in bull/bear markets by following the trend via Elder Ray.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6H_1D_ElderRay_BullBearPower_Trend_Follow"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Align to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6-period change in power for momentum confirmation
    bull_power_change = np.diff(bull_power_aligned, prepend=bull_power_aligned[0])
    bear_power_change = np.diff(bear_power_aligned, prepend=bear_power_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for calculations
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bull power positive AND rising + price above EMA13
            if bull_power_aligned[i] > 0 and bull_power_change[i] > 0 and close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bear power positive AND rising + price below EMA13
            elif bear_power_aligned[i] > 0 and bear_power_change[i] > 0 and close[i] < ema13_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative OR price below EMA13
            if bull_power_aligned[i] <= 0 or close[i] < ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns negative OR price above EMA13
            if bear_power_aligned[i] <= 0 or close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals