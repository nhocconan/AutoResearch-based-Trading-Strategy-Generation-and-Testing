#!/usr/bin/env python3
"""
6h_ElderRay_Energy_Index_With_WeeklyTrend_v1
Hypothesis: Use Elder Ray Index (Bull Power/Bear Power) on 6h to detect institutional buying/selling pressure, filtered by weekly trend direction. Bull Power > 0 and Bear Power < 0 indicates balanced momentum; we go long when Bull Power rises above its 13-period EMA and weekly trend is up, short when Bear Power falls below its 13-period EMA and weekly trend is down. This captures momentum shifts with trend filtering to avoid whipsaws in ranging markets. Designed for low trade frequency (<30/year) to minimize fee drag while capturing sustained moves in both bull and bear markets.
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
    
    # Elder Ray Index components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Signal lines: EMA of Bull/Bear Power
    bull_ema = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_ema = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need EMA13 and signal lines
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(bull_ema[i]) or
            np.isnan(bear_ema[i])):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        be = bull_ema[i]
        re = bear_ema[i]
        weekly_trend_up = ema34_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False  # placeholder, will be replaced properly below
        # Fix: Get current weekly EMA value properly
        weekly_ema_val = ema34_1w_aligned[i]
        # We need the actual weekly close value at this point - but we don't have it aligned
        # Instead, use the weekly EMA vs previous weekly EMA to determine trend
        if i >= 1:
            weekly_prev = ema34_1w_aligned[i-1]
            weekly_trend_up = weekly_ema_val > weekly_prev
            weekly_trend_down = weekly_ema_val < weekly_prev
        else:
            weekly_trend_up = False
            weekly_trend_down = False
        
        if position == 0:
            # Long: Bull Power rising above its EMA and weekly trend up
            if bp > be and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power falling below its EMA and weekly trend down
            elif br < re and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Bull Power falls below its EMA or weekly trend turns down
            if bp < be or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Bear Power rises above its EMA or weekly trend turns up
            if br > re or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Energy_Index_With_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0