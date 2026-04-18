#!/usr/bin/env python3
"""
12h_Pivot_R2_S2_Breakout_WeeklyTrend_Volume
Hypothesis: Price breaks above/below weekly R2/S2 levels with volume confirmation and 1w EMA34 trend filter.
Uses weekly pivot levels (R2,S2 from prior week), volume > 1.5x 20-period average, and 1w EMA34 trend filter.
Designed for 12h timeframe to target 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
Works in both bull and bear markets by requiring trend alignment and volume confirmation.
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
    volume = prices['volume'].values
    
    # Weekly OHLC for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels: R2 = close + 1.1*(high-low)/2, S2 = close - 1.1*(high-low)/2
    # Using weekly range for stronger levels
    weekly_range = high_1w - low_1w
    r2 = close_1w + 1.1 * weekly_range / 2
    s2 = close_1w - 1.1 * weekly_range / 2
    
    # Align to 12h timeframe (use previous week's levels)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: >1.5x 20-period average (adjusted for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above S2 with volume spike in uptrend
            if (price > s2_level and          # breaks above S2
                vol_spike and                 # volume confirmation
                price > ema34):               # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R2 with volume spike in downtrend
            elif (price < r2_level and        # breaks below R2
                  vol_spike and               # volume confirmation
                  price < ema34):             # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below S2 or trend reverses
            if price < s2_level or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above R2 or trend reverses
            if price > r2_level or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Pivot_R2_S2_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0