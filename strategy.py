#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with weekly trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (EMA13 on 6h closes).
# Long when Bull Power > 0 AND Bear Power < 0 AND weekly EMA50 trend up (close > weekly EMA50) AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND Bull Power < 0 AND weekly EMA50 trend down (close < weekly EMA50) AND volume > 1.5x 20-period average.
# Exit when Bull Power and Bear Power have same sign (both positive or both negative) indicating loss of momentum.
# Uses 6h timeframe with 1w EMA for trend filter. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    close_w = df_w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 50)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, weekly uptrend, volume filter
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema50_w_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power > 0, Bull Power < 0, weekly downtrend, volume filter
            short_cond = (bear_power[i] > 0) and (bull_power[i] < 0) and (close[i] < ema50_w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: loss of momentum (both powers positive or both negative)
            if (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: loss of momentum (both powers positive or both negative)
            if (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals