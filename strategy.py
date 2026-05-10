#!/usr/bin/env python3
"""
6h_ElderRay_13_EMA_Power_BullBear_12hTrend
Hypothesis: Elder Ray (bull/bear power) on 6m EMA13 with 12h EMA50 trend filter.
Bull power = high - EMA13, Bear power = EMA13 - low.
Long when bull power > 0, bear power < 0, and price > 12h EMA50.
Short when bear power > 0, bull power < 0, and price < 12h EMA50.
Designed to work in both bull and bear markets by following 12h trend.
Target: 20-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "6h_ElderRay_13_EMA_Power_BullBear_12hTrend"
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
    
    # Calculate EMA13 for Elder Ray
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, n):
            ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    # Calculate bull power and bear power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate 12h EMA50 for trend filter (using HTF data)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_50_12h[i-1]
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)  # Ensure EMA13 and EMA50 are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema13[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive, bear power negative, and uptrend
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive, bull power negative, and downtrend
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bear power becomes positive (trend weakening)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bull power becomes positive (trend weakening)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals