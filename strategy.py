#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_v1
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA34 trend filter.
- Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA34 (uptrend)
- Short when Bear Power < 0 AND Bull Power > 0 AND price < 1d EMA34 (downtrend)
- Uses 13-period EMA for power calculations (standard Elder Ray)
- Trend filter avoids counter-trend trades in strong markets
- Designed for low frequency (target: 50-150 total trades over 4 years) with strong BTC/ETH edge
- Works in both bull and bear markets by following 1d trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 13 for EMA13)
    start_idx = max(34, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray conditions with 1d trend filter
        bull_positive = bull_power[i] > 0
        bear_negative = bear_power[i] < 0
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND 1d uptrend
            if bull_positive and bear_negative and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND 1d downtrend
            elif bear_negative and bull_positive and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR Bear Power >= 0 OR 1d trend turns down
            if not (bull_positive and bear_negative) or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power >= 0 OR Bull Power <= 0 OR 1d trend turns up
            if not (bear_negative and bull_positive) or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0