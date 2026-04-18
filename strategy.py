#!/usr/bin/env python3
"""
1d Weekly Pivot Point Reversal with Volume Confirmation
Uses weekly pivot levels (R1/S1) as support/resistance for mean-reversion entries.
Long when price touches S1 with volume spike and closes above it.
Short when price touches R1 with volume spike and closes below it.
Weekly trend filter (price vs weekly EMA20) to avoid counter-trend trades.
Designed for low trade frequency with clear mean-reversion edge in ranging markets.
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
    
    # Get weekly data for pivot points and EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from prior week
    # Using weekly high, low, close from previous completed week
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (wk_high + wk_low + wk_close) / 3.0
    # R1 = 2*P - L
    r1 = 2 * pp - wk_low
    # S1 = 2*P - H
    s1 = 2 * pp - wk_high
    
    # Align weekly levels to daily (available after weekly bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA20 for trend filter
    wk_ema20 = pd.Series(wk_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    wk_ema20_aligned = align_htf_to_ltf(prices, df_1w, wk_ema20)
    
    # Volume spike detection (2x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(wk_ema20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_weekly_ema = price > wk_ema20_aligned[i]
        below_weekly_ema = price < wk_ema20_aligned[i]
        
        if position == 0:
            # Long: price touches or crosses above S1 with volume spike, in uptrend
            if (price >= s1_aligned[i] and volume_spike[i] and above_weekly_ema):
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses below R1 with volume spike, in downtrend
            elif (price <= r1_aligned[i] and volume_spike[i] and below_weekly_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until reversal signal or trend change
            signals[i] = 0.25
            # Exit: price crosses below S1 or trend turns down
            if price < s1_aligned[i] or below_weekly_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reversal signal or trend change
            signals[i] = -0.25
            # Exit: price crosses above R1 or trend turns up
            if price > r1_aligned[i] or above_weekly_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_VolumeSpike_WeeklyEMA20"
timeframe = "1d"
leverage = 1.0