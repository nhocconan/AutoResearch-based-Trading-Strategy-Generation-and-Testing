#!/usr/bin/env python3
"""
6h_Pivot_R2_S2_Breakout_Volume_Trend_Filter
Hypothesis: Price breaks above/below R2/S2 levels with volume confirmation and trend filter.
Uses 1d Camarilla pivot levels (R2/S2), volume > 2x 20-period average, and EMA20 trend filter.
Designed to work in both bull and bear markets by requiring trend alignment.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while capturing institutional breakout moves.
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
    
    # Daily Camarilla pivot levels (calculated from previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R2, S2 based on previous day
    # R2 = close + 1.1*(high-low)/6
    # S2 = close - 1.1*(high-low)/6
    camarilla_range = (high_1d - low_1d) * 1.1 / 6
    r2 = close_1d + camarilla_range
    s2 = close_1d - camarilla_range
    
    # Align to 6h timeframe (use previous day's levels for current day)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # EMA20 for trend filter on 6h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        ema20 = ema_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above S2 with volume spike in uptrend
            if (price > s2_level and          # breaks above S2
                vol_spike and                 # volume confirmation
                price > ema20):               # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R2 with volume spike in downtrend
            elif (price < r2_level and        # breaks below R2
                  vol_spike and               # volume confirmation
                  price < ema20):             # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below S2 or trend reverses
            if price < s2_level or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above R2 or trend reverses
            if price > r2_level or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Pivot_R2_S2_Breakout_Volume_Trend_Filter"
timeframe = "6h"
leverage = 1.0