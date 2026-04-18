#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + 1w Trend Filter
- Bull Power (High - EMA13) and Bear Power (EMA13 - Low) measure buying/selling pressure
- In strong weekly uptrend (price > weekly EMA34), only take long when Bull Power > 0 and rising
- In strong weekly downtrend (price < weekly EMA34), only take short when Bear Power > 0 and rising
- Weekly trend filter prevents counter-trend trades in strong trends
- Works in both bull and bear markets by adapting to weekly trend direction
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
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
    
    # Calculate EMA13 for Elder Ray
    if len(close) >= 13:
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema13 = np.full_like(close, np.nan)
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    if len(close_1w) >= 34:
        ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema34_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA to 6h
    ema34_1w_6h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not available
        if np.isnan(ema34_1w_6h[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema34_1w_6h[i]
        weekly_downtrend = close[i] < ema34_1w_6h[i]
        
        if position == 0:
            # Long conditions: weekly uptrend + bull power positive and rising
            if weekly_uptrend and bull_power[i] > 0 and i > start_idx and bull_power[i] > bull_power[i-1]:
                signals[i] = 0.25
                position = 1
            # Short conditions: weekly downtrend + bear power positive and rising
            elif weekly_downtrend and bear_power[i] > 0 and i > start_idx and bear_power[i] > bear_power[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down OR bull power turns negative
            if not weekly_uptrend or bull_power[i] <= 0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up OR bear power turns negative
            if not weekly_downtrend or bear_power[i] <= 0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0