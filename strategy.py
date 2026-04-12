#!/usr/bin/env python3
"""
6h_12h_ElderRay_Retest_v1
Hypothesis: On 6h timeframe, enter long when Elder Bull Power > 0 and price retraces to EMA(13) during uptrend (EMA(13) > EMA(34)); enter short when Bear Power < 0 and price retraces to EMA(13) during downtrend (EMA(13) < EMA(34)). Uses 12h EMA alignment for trend filter to avoid counter-trend trades. Works in bull via long retests and bear via short retests. Targets 15-25 trades/year by requiring both trend alignment and retest precision.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_ElderRay_Retest_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMAs for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Elder Bull/Bear Power: EMA13 - EMA34
    bull_power = ema13 - ema34
    bear_power = ema13 - ema34  # Same calculation, negative for bear
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMAs for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema13_12h = close_12h.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(ema34[i]) or 
            np.isnan(bull_power[i]) or np.isnan(ema13_12h_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter from 12h: uptrend if EMA13 > EMA34, downtrend if EMA13 < EMA34
        uptrend_12h = ema13_12h_aligned[i] > ema34_12h_aligned[i]
        downtrend_12h = ema13_12h_aligned[i] < ema34_12h_aligned[i]
        
        # Entry conditions: retest of EMA13 with Elder Ray confirmation
        long_entry = (bull_power[i] > 0) and uptrend_12h and (close[i] >= ema13[i] * 0.998) and (close[i] <= ema13[i] * 1.002)
        short_entry = (bear_power[i] < 0) and downtrend_12h and (close[i] >= ema13[i] * 0.998) and (close[i] <= ema13[i] * 1.002)
        
        # Exit conditions: trend reversal or price moves >0.5% from EMA13
        long_exit = (not uptrend_12h) or (close[i] < ema13[i] * 0.995) or (close[i] > ema13[i] * 1.005)
        short_exit = (not downtrend_12h) or (close[i] > ema13[i] * 1.005) or (close[i] < ema13[i] * 0.995)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals