#!/usr/bin/env python3
# 6h_ElderRay_BullPower_BearPower_12hTrend_Filter
# Hypothesis: Elder Ray indicator (bull power = high - EMA13, bear power = EMA13 - low) on 6h with 12h EMA trend filter.
# Long when bull power > 0 and rising AND 12h trend up.
# Short when bear power < 0 and falling AND 12h trend down.
# Uses Elder Ray to measure bull/bear strength relative to EMA, with trend filter to avoid counter-trend whipsaw.
# Target: 50-150 trades over 4 years (~12-37/year) with controlled risk.

name = "6h_ElderRay_BullPower_BearPower_12hTrend_Filter"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        for i in range(34, len(close_12h)):
            ema34_12h[i] = (close_12h[i] * 2 + ema34_12h[i-1] * 32) / 34
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13 = np.full_like(close, np.nan)
    if len(close) >= 13:
        ema13[12] = np.mean(close[0:13])
        for i in range(13, len(close)):
            ema13[i] = (close[i] * 2 + ema13[i-1] * 11) / 13
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # high - EMA13
    bear_power = ema13 - low   # EMA13 - low
    
    # Align 12h indicators to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13)  # Need 12h EMA and 6h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close[i] > ema34_12h_aligned[i]
        
        if position == 0:
            # Enter long: bull power positive AND rising AND 12h trend up
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short: bear power positive AND rising AND 12h trend down
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and not trend_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative OR 12h trend turns down
            if bull_power[i] <= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns negative OR 12h trend turns up
            if bear_power[i] <= 0 or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals