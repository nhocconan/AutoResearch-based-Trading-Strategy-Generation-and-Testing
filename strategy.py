#!/usr/bin/env python3
"""
6h_ElderRay_With_1D_Trend_Filter
Hypothesis: Elder Ray (Bull/Bear Power) on 6h with daily EMA200 trend filter.
Long when Bull Power > 0, Bear Power < 0, and price > daily EMA200.
Short when Bull Power < 0, Bear Power > 0, and price < daily EMA200.
Uses EMA13 for power calculation. Trend filter avoids counter-trend trades in bear markets.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
Works in bull/bear: daily trend filter ensures trades align with higher timeframe direction.
"""

name = "6h_ElderRay_With_1D_Trend_Filter"
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
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_daily = df_daily['close'].values
    ema200_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 200:
        multiplier = 2.0 / (200 + 1)
        ema200_daily[199] = np.mean(close_daily[:200])
        for i in range(200, len(close_daily)):
            ema200_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema200_daily[i-1]
    ema200_daily_aligned = align_htf_to_ltf(prices, df_daily, ema200_daily)
    
    # Calculate EMA13 for Elder Ray (13-period EMA of close)
    ema13 = np.full(n, np.nan)
    if n >= 13:
        multiplier = 2.0 / (13 + 1)
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = multiplier * close[i] + (1 - multiplier) * ema13[i-1]
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Ensure EMA13 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema200_daily_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, and price > daily EMA200
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema200_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, and price < daily EMA200
            elif bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema200_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR price < daily EMA200
            if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema200_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power >= 0 OR Bear Power <= 0 OR price > daily EMA200
            if bull_power[i] >= 0 or bear_power[i] <= 0 or close[i] > ema200_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals