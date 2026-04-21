#!/usr/bin/env python3
"""
6h_1d_1w_ElderRay_Breakout_TrendFilter_v1
Hypothesis: Elder Ray (Bull/Bear Power) breakout with weekly trend filter and volume confirmation.
Long when Bull Power crosses above zero + weekly uptrend + volume spike.
Short when Bear Power crosses below zero + weekly downtrend + volume spike.
Exit when opposite power crosses zero or price reverts to EMA20.
Works in bull/bear by following weekly trend while capturing mean-reversion within trend.
Target: 15-30 trades/year per symbol (~60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for EMA13 (standard Elder Ray uses 13-period EMA)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan
    
    # Calculate 13-period EMA of previous close
    close_series = pd.Series(prev_close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    # Load weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter: EMA34 slope
        if i >= 51:
            ema34_prev = ema34_1w_aligned[i-1]
            ema34_curr = ema34_1w_aligned[i]
            weekly_uptrend = ema34_curr > ema34_prev
            weekly_downtrend = ema34_curr < ema34_prev
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long conditions: Bull Power crosses above zero + weekly uptrend + volume
            if bull_power_aligned[i] > 0 and bull_power_aligned[i-1] <= 0 and weekly_uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power crosses below zero + weekly downtrend + volume
            elif bear_power_aligned[i] < 0 and bear_power_aligned[i-1] >= 0 and weekly_downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear Power crosses above zero OR price crosses below EMA13
            if bear_power_aligned[i] > 0 or price < ema13_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power crosses below zero OR price crosses above EMA13
            if bull_power_aligned[i] < 0 or price > ema13_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_ElderRay_Breakout_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0