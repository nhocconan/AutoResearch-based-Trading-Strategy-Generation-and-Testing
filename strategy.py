#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_Trend_v2
Hypothesis: Weekly pivot-based directional bias with 6h Donchian breakouts and volume confirmation. 
Weekly pivots provide multi-day structure that works in both bull (continuation) and bear (reversal) markets.
Donchian(20) breakouts capture momentum, volume filter ensures commitment, and weekly trend filter avoids counter-trend trades.
Target: 15-35 trades/year per symbol (60-140 total over 4 years) with selective entries.
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
    volume = prices['volume'].values
    
    # Calculate weekly pivot levels (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and key levels (using standard pivot formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    # Key levels: R1, S1 for reversal zones; R2, S2 for stronger moves
    r1_1w = pivot_1w + range_1w * 1.1 / 12
    s1_1w = pivot_1w - range_1w * 1.1 / 12
    r2_1w = pivot_1w + range_1w * 1.1 / 6
    s2_1w = pivot_1w - range_1w * 1.1 / 6
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Weekly trend filter: price above/below weekly pivot
    weekly_uptrend = close > pivot_1w_aligned
    weekly_downtrend = close < pivot_1w_aligned
    
    # Donchian channel (20-period) on 6h timeframe
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA need 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high with volume and weekly uptrend bias
            # OR price rejects S1 support with volume and weekly uptrend
            if ((close[i] > donchian_high[i] and vol_spike[i] and weekly_uptrend[i]) or
                (close[i] > s1_1w_aligned[i] and low[i] <= s1_1w_aligned[i] and vol_spike[i] and weekly_uptrend[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low with volume and weekly downtrend bias
            # OR price rejects R1 resistance with volume and weekly downtrend
            elif ((close[i] < donchian_low[i] and vol_spike[i] and weekly_downtrend[i]) or
                  (close[i] < r1_1w_aligned[i] and high[i] >= r1_1w_aligned[i] and vol_spike[i] and weekly_downtrend[i])):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR weekly trend turns down
            if (close[i] < donchian_low[i] or not weekly_uptrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR weekly trend turns up
            if (close[i] > donchian_high[i] or not weekly_downtrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_Trend_v2"
timeframe = "6h"
leverage = 1.0