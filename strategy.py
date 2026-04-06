#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with weekly trend filter
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Enter long when Bull Power > 0 and weekly EMA(40) rising
# Enter short when Bear Power < 0 and weekly EMA(40) falling
# Exit when Elder Power reverses sign or weekly trend changes
# Target: 50-150 trades over 4 years by combining Elder Ray with weekly trend filter

name = "6h_elder_ray_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(13) for Elder Ray calculation on 6h
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Weekly EMA(40) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Weekly EMA slope (rising/falling)
    ema40_slope = np.diff(ema40_1w_aligned, prepend=ema40_1w_aligned[0])
    weekly_rising = ema40_slope > 0
    weekly_falling = ema40_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Wait for EMA13 to stabilize
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema40_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bull Power <= 0 OR weekly trend turns falling
            if bull_power[i] <= 0 or weekly_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power >= 0 OR weekly trend turns rising
            if bear_power[i] >= 0 or weekly_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Power extreme + weekly trend alignment
            if bull_power[i] > 0 and weekly_rising[i]:
                # Strong bullish momentum with rising weekly trend
                signals[i] = 0.25
                position = 1
            elif bear_power[i] < 0 and weekly_falling[i]:
                # Strong bearish momentum with falling weekly trend
                signals[i] = -0.25
                position = -1
    
    return signals