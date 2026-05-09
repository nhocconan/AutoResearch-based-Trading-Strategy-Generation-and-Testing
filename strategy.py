#!/usr/bin/env python3
# 6H_1D_ElderRay_BullBearPower_Trend_Follow
# Hypothesis: Use Elder Ray (Bull/Bear Power) on 1d with trend filter from 12h EMA50 on 6h timeframe.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Go long when Bull Power > 0 and Bear Power > 0 (both positive) with 12h EMA50 uptrend.
# Go short when Bull Power < 0 and Bear Power < 0 (both negative) with 12h EMA50 downtrend.
# Exit when power signs diverge or trend fails.
# Works in bull/bear by following higher timeframe trend, avoiding counter-trend whipsaw.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).

name = "6H_1D_ElderRay_BullBearPower_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: EMA50 rising/falling (use 3-period slope)
    ema50_slope = pd.Series(ema50_12h).diff(3).values
    ema50_up = ema50_slope > 0
    ema50_down = ema50_slope < 0
    
    # Align indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_up_aligned = align_htf_to_ltf(prices, df_12h, ema50_up)
    ema50_down_aligned = align_htf_to_ltf(prices, df_12h, ema50_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(ema50_up_aligned[i]) or np.isnan(ema50_down_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: both powers positive + 12h EMA50 uptrend
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] > 0 and ema50_up_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: both powers negative + 12h EMA50 downtrend
            elif bull_power_aligned[i] < 0 and bear_power_aligned[i] < 0 and ema50_down_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: powers diverge or trend fails
            if not (bull_power_aligned[i] > 0 and bear_power_aligned[i] > 0) or not ema50_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: powers diverge or trend fails
            if not (bull_power_aligned[i] < 0 and bear_power_aligned[i] < 0) or not ema50_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals