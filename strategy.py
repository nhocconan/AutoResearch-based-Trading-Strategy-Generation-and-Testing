#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_1wTrend_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) zero-cross with 1w trend filter.
- Long when 6h Bull Power crosses above zero AND 1w close > 1w EMA34 (uptrend)
- Short when 6h Bear Power crosses below zero AND 1w close < 1w EMA34 (downtrend)
- Elder Ray = Bull Power = High - EMA13, Bear Power = Low - EMA13 (EMA13 on 6h close)
- Uses 1w EMA34 for higher timeframe trend to avoid counter-trend whipsaws
- Zero-cross ensures momentum shift, reducing false signals
- Designed for low frequency (target 12-30 trades/year) to minimize fee drag
- Exit on opposite Elder Ray zero-cross or trend reversal
- Novelty: Combines Elder Ray momentum with weekly trend filter for BTC/ETH edge in both bull/bear markets
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
    
    # Load 1w data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter (needs completed 1w candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1w = np.where(ema_34_1w_aligned > 0, 
                        np.where(close > ema_34_1w_aligned, 1, -1), 
                        0)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Zero-cross signals: previous value <= 0 and current > 0 (bull), or previous >= 0 and current < 0 (bear)
    bull_cross = (bull_power[:-1] <= 0) & (bull_power[1:] > 0)
    bear_cross = (bear_power[:-1] >= 0) & (bear_power[1:] < 0)
    # Shift to align with current bar (signal at i based on cross between i-1 and i)
    bull_cross = np.append(False, bull_cross)
    bear_cross = np.append(False, bear_cross)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 13 for 6h EMA)
    start_idx = max(34, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray zero-cross with 1w trend filter
        if position == 0:
            # Long: Bull Power crosses above zero AND 1w uptrend
            if bull_cross[i] and trend_1w[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses below zero AND 1w downtrend
            elif bear_cross[i] and trend_1w[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bear Power crosses below zero OR 1w trend turns down
            if bear_cross[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power crosses above zero OR 1w trend turns up
            if bull_cross[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_1wTrend_v1"
timeframe = "6h"
leverage = 1.0