#!/usr/bin/env python3
# 6H_ElderRay_1dTrend_Filter
# Hypothesis: Uses Elder Ray (Bull/Bear Power) on 6h chart filtered by 1-day EMA13 trend.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Enters long when Bull Power > 0 and rising
# in uptrend (close > EMA13). Enters short when Bear Power > 0 and rising in downtrend (close < EMA13).
# Exits when power becomes negative or trend reverses. Uses 13-period for responsiveness.
# Targets 15-35 trades per year on 6h timeframe with position size 0.25.

name = "6H_ElderRay_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for trend (EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for trend direction
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate EMA13 on 6h for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # High - EMA13
    bear_power = ema_13_6h - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for EMA13
    
    for i in range(start_idx, n):
        if np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA13
        price_above_ema = close[i] > ema_13_1d_aligned[i]
        price_below_ema = close[i] < ema_13_1d_aligned[i]
        
        # Elder Ray signals with momentum (rising power)
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_rising = bear_power[i] > bear_power[i-1]
        
        if position == 0:
            # Long entry: Bull Power > 0 and rising in uptrend
            if (bull_power[i] > 0 and 
                bull_rising and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 and rising in downtrend
            elif (bear_power[i] > 0 and 
                  bear_rising and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power becomes negative or trend reverses
            if (bull_power[i] <= 0 or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power becomes negative or trend reverses
            if (bear_power[i] <= 0 or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals