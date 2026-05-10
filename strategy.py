#!/usr/bin/env python3
# 6H_ElderRay_ForceIndex_Combo
# Hypothesis: Combines Elder Ray (Bull/Bear Power) with Force Index on 6h timeframe,
# using 1d EMA13 for trend filter. Enters long when Bull Power > 0 and Force Index > 0 in uptrend (close > EMA13).
# Enters short when Bear Power < 0 and Force Index < 0 in downtrend (close < EMA13).
# Uses volume-weighted confirmation to avoid whipsaws. Designed for 6h timeframe to target 12-37 trades/year.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "6H_ElderRay_ForceIndex_Combo"
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
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for trend direction
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate Force Index: (Close - Close_prev) * Volume
    # Using 1-period force index
    close_shift = np.roll(close, 1)
    close_shift[0] = 0  # avoid using future data
    force_index = (close - close_shift) * volume
    # Smooth Force Index with EMA(3) to reduce noise
    force_index_smooth = pd.Series(force_index).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Warmup for EMA and Force Index
    
    for i in range(start_idx, n):
        if np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(force_index_smooth[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA13
        price_above_ema = close[i] > ema_13_1d_aligned[i]
        price_below_ema = close[i] < ema_13_1d_aligned[i]
        
        if position == 0:
            # Long entry: Bull Power > 0 AND Force Index > 0 in uptrend
            if (bull_power[i] > 0 and 
                force_index_smooth[i] > 0 and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 AND Force Index < 0 in downtrend
            elif (bear_power[i] < 0 and 
                  force_index_smooth[i] < 0 and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR Force Index <= 0 OR trend reverses
            if (bull_power[i] <= 0 or 
                force_index_smooth[i] <= 0 or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR Force Index >= 0 OR trend reverses
            if (bear_power[i] >= 0 or 
                force_index_smooth[i] >= 0 or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals