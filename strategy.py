#!/usr/bin/env python3
"""
6h Bull/Bear Power + EMA200 Trend
Hypothesis: Elder Ray's Bull Power (High - EMA) and Bear Power (Low - EMA) capture institutional buying/selling pressure.
Combined with EMA200 trend filter, this identifies strong momentum in trending markets while avoiding false signals in ranges.
Works in bull/bear because EMA200 adapts to trend direction and Bull/Bear Power confirms momentum behind moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Bull Power and Bear Power on 6h data
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(ema13[i]):
            signals[i] = 0.0
            continue
        
        trend = ema200_1d_aligned[i]
        bp = bull_power[i]
        bp_prev = bull_power[i-1] if i > 0 else 0
        br = bear_power[i]
        br_prev = bear_power[i-1] if i > 0 else 0
        
        if position == 0:
            # Enter long when Bull Power turns positive in uptrend
            if bp > 0 and bp_prev <= 0 and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short when Bear Power turns negative in downtrend
            elif br < 0 and br_prev >= 0 and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Bull Power turns negative or trend fails
            if bp <= 0 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Bear Power turns positive or trend fails
            if br >= 0 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BullBearPower_EMA200_Trend"
timeframe = "6h"
leverage = 1.0