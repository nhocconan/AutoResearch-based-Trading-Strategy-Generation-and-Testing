#!/usr/bin/env python3
"""
4H_BullBear_Pivot_Breakout_v1
Hypothesis: Use daily pivot levels as structural support/resistance with 4h trend filter (EMA34) and volume confirmation.
In bull markets: buy breakouts above daily R1 when price above 4h EMA34 and volume > 1.5x average.
In bear markets: sell breakdowns below daily S1 when price below 4h EMA34 and volume > 1.5x average.
This structure-based approach works in both regimes by using pivots as dynamic S/R and EMA34 for trend alignment.
Targets 30-60 trades/year on 4h timeframe with discrete sizing (0.25) to minimize fee drag.
"""
name = "4H_BullBear_Pivot_Breakout_v1"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Get 4h EMA34 for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d      # Resistance 1
    s1 = 2 * pivot - high_1d     # Support 1
    
    # Align daily pivots to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current 4h volume > 1.5 x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 4h EMA34 (uptrend), 4h close above daily R1, volume confirmation
            if (close[i] > ema_34[i] and 
                close[i] > r1_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 4h EMA34 (downtrend), 4h close below daily S1, volume confirmation
            elif (close[i] < ema_34[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA34 (trend change)
            if close[i] < ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 4h EMA34 (trend change)
            if close[i] > ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals