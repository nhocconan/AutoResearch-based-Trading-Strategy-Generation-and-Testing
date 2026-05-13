#!/usr/bin/env python3
"""
6h_ElderRay_WeeklyTrend
Hypothesis: Elder Ray (Bull/Bear Power) identifies market strength relative to EMA. Combined with weekly trend filter (price above/below weekly EMA20), it captures strong trends while avoiding counter-trend noise. Works in bull markets by catching sustained uptrends and in bear markets by catching sharp reversals with institutional participation. Volume confirmation ensures validity.
"""

name = "6h_ElderRay_WeeklyTrend"
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
    
    # Calculate EMA22 for Elder Ray
    ema_period = 22
    close_s = pd.Series(close)
    ema = close_s.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Elder Ray components
    bull_power = high - ema
    bear_power = low - ema
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > ema20_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    # Calculate volume average (50-period) for volume filter
    vol_ma_50 = np.zeros_like(volume)
    for i in range(49, len(volume)):
        vol_ma_50[i] = np.mean(volume[i-49:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 50-period average
        vol_filter = volume[i] > 1.5 * vol_ma_50[i]
        
        if position == 0:
            # LONG: Positive Bull Power + weekly uptrend + volume
            if (bull_power[i] > 0 and weekly_uptrend_aligned[i] and vol_filter):
                signals[i] = 0.25
                position = 1
            # SHORT: Negative Bear Power + weekly downtrend + volume
            elif (bear_power[i] < 0 and not weekly_uptrend_aligned[i] and vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power turns positive (selling pressure) or loss of weekly uptrend
            if (bear_power[i] >= 0 or not weekly_uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power turns negative (buying pressure) or gain of weekly uptrend
            if (bull_power[i] <= 0 or weekly_uptrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals