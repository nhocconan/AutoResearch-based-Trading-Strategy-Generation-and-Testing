#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_v1
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA trend filter.
- Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Long when Bull Power > 0 AND Bear Power rising AND close > 1d EMA50
- Short when Bear Power < 0 AND Bull Power falling AND close < 1d EMA50
- Uses 6h timeframe for lower trade frequency (target: 50-150 total trades over 4 years)
- 1d EMA50 ensures alignment with higher timeframe trend
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by combining momentum (Elder Ray) with trend filter
"""

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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 13 for EMA13, 50 for 1d EMA50)
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if 1d EMA50 not ready
        if np.isnan(ema50_1d_aligned[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray conditions (require previous bar for rising/falling)
        if i < start_idx + 1:
            signals[i] = 0.0
            continue
            
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 AND rising AND close > 1d EMA50
            if bull_power[i] > 0 and bull_rising and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND falling AND close < 1d EMA50
            elif bear_power[i] < 0 and bear_falling and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR close < 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power >= 0 OR close > 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_v1"
timeframe = "6h"
leverage = 1.0