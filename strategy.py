#!/usr/bin/env python3
"""
6h_ElderRay_Energy_Index_With_WeeklyTrend_v1
Hypothesis: Use Elder Ray (Bull/Bear Power) for mean reversion signals aligned with weekly trend. 
- Long: Bear Power crosses above zero (bullish momentum emerging) AND weekly close > weekly EMA20 (bullish regime)
- Short: Bull Power crosses below zero (bearish momentum emerging) AND weekly close < weekly EMA20 (bearish regime)
- Exit when power crosses back through zero or weekly trend flips.
Elder Ray identifies momentum shifts early; weekly trend filter ensures alignment with higher-timeframe bias. 
Designed for low trade frequency (~15-25/year) to minimize fee drift while capturing momentum reversals in both bull and bear markets.
"""

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
    
    # Weekly trend filter: EMA20 on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need EMA13 and weekly alignment
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_ema_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema13[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend_up = close[i] > weekly_ema_aligned[i]
        weekly_trend_down = close[i] < weekly_ema_aligned[i]
        bp = bear_power[i]
        bp_prev = bear_power[i-1]
        bp2 = bull_power[i]
        bp2_prev = bull_power[i-1]
        
        if position == 0:
            # Long: Bear Power crosses above zero (from negative to positive) in bullish weekly regime
            if bp_prev <= 0 and bp > 0 and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power crosses below zero (from positive to negative) in bearish weekly regime
            elif bp2_prev >= 0 and bp2 < 0 and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Bear Power crosses back below zero OR weekly trend turns bearish
            if bp < 0 or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Bull Power crosses back above zero OR weekly trend turns bullish
            if bp2 > 0 or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Energy_Index_With_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0