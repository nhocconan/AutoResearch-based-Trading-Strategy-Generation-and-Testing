#!/usr/bin/env python3
"""
4h_4h1d_Triple_EMA_Crossover_Trend_Follow_v1
Hypothesis: Use a fast EMA (9), medium EMA (21), and slow EMA (50) on 4h timeframe.
Enter long when EMA9 > EMA21 > EMA50 (bullish alignment), short when EMA9 < EMA21 < EMA50 (bearish alignment).
Exit when the fast EMA crosses back over/under the medium EMA. This captures trends while avoiding whipsaws.
Works in both bull and bear markets because it follows the trend direction.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate EMAs on 4h close prices
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any EMA data is NaN
        if np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema50[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Bullish alignment: EMA9 > EMA21 > EMA50
            if ema9[i] > ema21[i] and ema21[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Bearish alignment: EMA9 < EMA21 < EMA50
            elif ema9[i] < ema21[i] and ema21[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: EMA9 crosses EMA21
        elif position == 1:
            # Exit long: EMA9 crosses below EMA21
            if ema9[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA9 crosses above EMA21
            if ema9[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4h1d_Triple_EMA_Crossover_Trend_Follow_v1"
timeframe = "4h"
leverage = 1.0