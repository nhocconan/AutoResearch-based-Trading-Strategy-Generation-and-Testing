#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend
# Hypothesis: Elder Ray index (Bull Power = High - EMA13, Bear Power = Low - EMA13) with 1d EMA50 trend filter.
# In uptrend (price > 1d EMA50), enter long when Bull Power > 0 and rising; enter short when Bear Power < 0 and falling.
# In downtrend (price < 1d EMA50), enter short when Bear Power < 0 and falling; enter long when Bull Power > 0 and rising.
# Uses 13-period EMA for Elder Ray calculation. Designed for 15-30 trades/year on 6h timeframe.

name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) with proper initialization
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray
    ema_13 = np.full_like(close, np.nan)
    if len(close) >= 13:
        ema_13[12] = np.mean(close[0:13])
        for i in range(13, len(close)):
            ema_13[i] = (close[i] * 2 + ema_13[i-1] * 11) / 13
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising (bull_power[i] > bull_power[i-1]) AND uptrend (price > 1d EMA50)
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 and falling (bear_power[i] < bear_power[i-1]) AND downtrend (price < 1d EMA50)
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 or falling OR trend turns bearish
            if bull_power[i] <= 0 or bull_power[i] < bull_power[i-1] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 or rising OR trend turns bullish
            if bear_power[i] >= 0 or bear_power[i] > bear_power[i-1] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals