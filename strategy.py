#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_With_1dTrend_Filter
Hypothesis: Elder Ray (Bull/Bear Power) identifies bullish/bearish momentum via EMA13 divergence. 
Trade in direction of 1d EMA34 trend: long when Bull Power > 0 and Bear Power < 0 in uptrend; 
short when Bear Power < 0 and Bull Power > 0 in downtrend. Uses 6h timeframe for ~20-40 trades/year.
"""

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
    
    # 6-day EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Warmup for EMA13
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            # Long: Bull Power > 0 and Bear Power < 0 in uptrend
            if bull > 0 and bear < 0 and ema_trend > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and Bull Power > 0 in downtrend
            elif bear < 0 and bull > 0 and ema_trend < close[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if momentum fades or trend reverses
            if bull <= 0 or bear >= 0 or ema_trend < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if momentum fades or trend reverses
            if bear >= 0 or bull <= 0 or ema_trend > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_With_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0