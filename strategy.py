#!/usr/bin/env python3
"""
6h_ElderRay_2Period_Trend_Filter
Hypothesis: Elder Ray (Bull Power/Bear Power) uses EMA13 trend and price extremes to capture trend strength.
Long when Bull Power > 0 and rising; Short when Bear Power < 0 and falling.
Works in bull/bear by following the trend defined by EMA13. Uses 12h EMA200 as higher timeframe filter to avoid counter-trend trades.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "6h_ElderRay_2Period_Trend_Filter"
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
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, n):
            ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA200 on 12h for higher timeframe trend filter
    ema200_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 200:
        ema200_12h[199] = np.mean(close_12h[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_12h)):
            ema200_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema200_12h[i-1]
    
    # Align 12h EMA200 to 6h
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA200
        uptrend_12h = close[i] > ema200_12h_aligned[i]
        downtrend_12h = close[i] < ema200_12h_aligned[i]
        
        # Elder Ray conditions with slope (current > previous)
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, in uptrend
            if bull_power[i] > 0 and bull_power_rising and uptrend_12h:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, in downtrend
            elif bear_power[i] < 0 and bear_power_falling and downtrend_12h:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 or not rising or trend turns down
            if bull_power[i] <= 0 or not bull_power_rising or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 or not falling or trend turns up
            if bear_power[i] >= 0 or not bear_power_falling or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals