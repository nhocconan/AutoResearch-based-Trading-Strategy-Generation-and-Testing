#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_SMA200_Trend_v2
Hypothesis: Elder Ray (Bull Power/Bear Power) + 200-period SMA trend filter on 1d timeframe.
Go long when Bull Power > 0 and price > 1d SMA200, short when Bear Power < 0 and price < 1d SMA200.
Uses 13-period EMA for Bull/Bear Power calculation. Designed to capture trends with clear momentum,
avoiding whipsaws in sideways markets. Targets 20-50 trades/year on 6h timeframe.
"""

name = "6h_ElderRay_BullBearPower_SMA200_Trend_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA13 (Elder Ray) and SMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema_13_1d
    
    # Calculate 200-period SMA for trend filter
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(sma_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 and price > 1d SMA200
            if bull_power_aligned[i] > 0 and close[i] > sma_200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 and price < 1d SMA200
            elif bear_power_aligned[i] < 0 and close[i] < sma_200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or price <= 1d SMA200
            if bull_power_aligned[i] <= 0 or close[i] <= sma_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 or price >= 1d SMA200
            if bear_power_aligned[i] >= 0 or close[i] >= sma_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals