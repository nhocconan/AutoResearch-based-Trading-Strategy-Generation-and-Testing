#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with weekly trend filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and weekly EMA(34) uptrend
# Short when Bear Power > 0 and weekly EMA(34) downtrend
# Elder Ray measures bull/bear strength relative to trend.
# Weekly trend filter ensures we trade with higher timeframe momentum.
# This combination works in both bull and bear markets by adapting to trend.

name = "6h_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Elder Ray components: Bull Power and Bear Power
    # Bull Power = High - EMA(13)
    # Bear Power = EMA(13) - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        if position == 0:
            # Enter long: Bull Power positive (bulls in control) + weekly uptrend
            if bp > 0 and close[i] > ema34_1w_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power positive (bears in control) + weekly downtrend
            elif br > 0 and close[i] < ema34_1w_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power becomes positive OR weekly trend turns down
            if br > 0 or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power becomes positive OR weekly trend turns up
            if bp > 0 or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals