#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Trend Filter
# Uses 6h Elder Ray (Bull/Bear Power) for momentum and 12h EMA200 for trend direction.
# Long when Bull Power > 0 and price > 12h EMA200; Short when Bear Power < 0 and price < 12h EMA200.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA200
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate 6h Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if EMA200 data not ready
        if np.isnan(ema200_12h_aligned[i]):
            continue
        
        # Long entry: Bull Power > 0 (bullish momentum) and price above 12h EMA200 (uptrend)
        if (bull_power[i] > 0 and
            close[i] > ema200_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power < 0 (bearish momentum) and price below 12h EMA200 (downtrend)
        elif (bear_power[i] < 0 and
              close[i] < ema200_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: momentum reversal or trend change
        elif position == 1 and (bull_power[i] <= 0 or close[i] <= ema200_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power[i] >= 0 or close[i] >= ema200_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA200_Trend"
timeframe = "6h"
leverage = 1.0