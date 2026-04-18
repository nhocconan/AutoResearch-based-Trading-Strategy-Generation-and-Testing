#!/usr/bin/env python3
"""
4h_MultiTF_PivotBreakout_VolumeFilter
Hypothesis: Price breaks above/below key pivot levels from higher timeframe (1d) with volume confirmation and EMA trend filter.
Uses daily Camarilla pivot levels (R1, S1) to identify institutional support/resistance.
Requires volume > 1.8x 20-period average and EMA20 trend alignment.
Designed to capture institutional breakouts in both bull and bear markets with tight entry conditions.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag while capturing high-probability moves.
"""

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
    volume = prices['volume'].values
    
    # Daily data for pivot levels (HIGHER TIMEFRAME)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Standard formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day (based on previous day)
    pivot_range = (high_1d - low_1d) * 1.1 / 12
    r1_levels = close_1d + pivot_range  # Resistance 1
    s1_levels = close_1d - pivot_range  # Support 1
    
    # Align to 4h timeframe with proper delay (use previous day's levels)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_levels)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_levels)
    
    # Volume filter: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # EMA20 trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_4h[i]
        s1 = s1_4h[i]
        vol_ok = volume_filter[i]
        ema20 = ema_20[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or trend reverses
            if price < s1 or price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or trend reverses
            if price > r1 or price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_MultiTF_PivotBreakout_VolumeFilter"
timeframe = "4h"
leverage = 1.0