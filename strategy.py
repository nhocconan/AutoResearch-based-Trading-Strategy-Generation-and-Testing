#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter
Hypothesis: Uses 1-day EMA200 as trend filter and 6-hour Elder Ray (Bull Power/Bear Power) for entry.
In bull trends (price > EMA200), go long when Bear Power crosses above zero (selling pressure weakening).
In bear trends (price < EMA200), go short when Bull Power crosses below zero (buying pressure weakening).
This captures mean-reversion within the trend, reducing false signals in strong trends.
Targets 15-25 trades/year per symbol. Works in both bull (2021, 2023-24) and bear (2022) markets.
"""

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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[0:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema200_1d[i] = close_1d[i] * alpha + ema200_1d[i-1] * (1 - alpha)
    
    # Align 1d EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6-day EMA13 for Elder Ray (using 6h data, 6 periods = 1 day)
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[0:13])
        alpha = 2 / (13 + 1)
        for i in range(13, n):
            ema13[i] = close[i] * alpha + ema13[i-1] * (1 - alpha)
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 13)  # Need EMA200 and EMA13 ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA200
        is_bull_trend = close[i] > ema200_aligned[i]
        is_bear_trend = close[i] < ema200_aligned[i]
        
        if position == 0:
            # Long: in bull trend, Bear Power crosses above zero (selling pressure weakening)
            if is_bull_trend and bear_power[i] > 0 and bear_power[i-1] <= 0:
                signals[i] = 0.25
                position = 1
            # Short: in bear trend, Bull Power crosses below zero (buying pressure weakening)
            elif is_bear_trend and bull_power[i] < 0 and bull_power[i-1] >= 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend turns bearish or Bear Power goes negative (selling pressure increases)
            if not is_bull_trend or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend turns bullish or Bull Power goes positive (buying pressure increases)
            if not is_bear_trend or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter"
timeframe = "6h"
leverage = 1.0