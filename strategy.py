#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d Trend Filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# 1d EMA(50) as trend filter: only take long when price > EMA50, short when price < EMA50
# Enter when Bull Power > 0 (for long) or Bear Power < 0 (for short) with trend alignment
# Exit when power crosses zero or trend changes
# Designed to capture momentum in trending markets while avoiding counter-trend trades
# Target: 15-25 trades/year to minimize fee drag
name = "6h_ElderRay_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6-period EMA(13) for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema_13[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Bull Power > 0 AND uptrend
            if bull_power[i] > 0 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 AND downtrend
            elif bear_power[i] < 0 and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power <= 0 OR trend turns down
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power >= 0 OR trend turns up
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals