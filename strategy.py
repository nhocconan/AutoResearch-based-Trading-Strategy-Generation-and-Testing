#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with weekly trend filter
# - Bull Power = High - EMA13(Close) measures bull strength
# - Bear Power = EMA13(Close) - Low measures bear strength  
# - Long when Bull Power > 0 AND Bear Power < 0 AND weekly EMA20 rising
# - Short when Bear Power > 0 AND Bull Power < 0 AND weekly EMA20 falling
# - Exit when power signals reverse or weekly trend changes
# - Uses 13-period EMA for sensitivity, weekly trend for direction filter
# - Target: 12-25 trades per year per symbol (48-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA20 for trend direction
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_prev = np.roll(ema20_weekly, 1)
    ema20_weekly_prev[0] = ema20_weekly[0]
    weekly_uptrend = ema20_weekly > ema20_weekly_prev
    weekly_downtrend = ema20_weekly < ema20_weekly_prev
    weekly_uptrend = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float))
    weekly_downtrend = align_htf_to_ltf(prices, df_weekly, weekly_downtrend.astype(float))
    
    # Calculate 13-period EMA for Elder Ray
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power
    high = prices['high'].values
    low = prices['low'].values
    
    bull_power = high - ema13  # High - EMA
    bear_power = ema13 - low   # EMA - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(weekly_uptrend[i]) or np.isnan(weekly_downtrend[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power positive AND Bear Power negative AND weekly uptrend
            if bull_power[i] > 0 and bear_power[i] < 0 and weekly_uptrend[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power positive AND Bull Power negative AND weekly downtrend
            elif bear_power[i] > 0 and bull_power[i] < 0 and weekly_downtrend[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Power signals reverse OR weekly trend turns down
            if bull_power[i] <= 0 or bear_power[i] >= 0 or weekly_downtrend[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Power signals reverse OR weekly trend turns up
            if bear_power[i] <= 0 or bull_power[i] >= 0 or weekly_uptrend[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_WeeklyTrend"
timeframe = "6h"
leverage = 1.0