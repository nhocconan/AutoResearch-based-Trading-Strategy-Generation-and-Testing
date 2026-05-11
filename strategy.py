#!/usr/bin/env python3
# 1d_WeeklyPivot_Reversal_1wTrend
# Hypothesis: Uses weekly pivot point reversal signals on daily timeframe, filtered by weekly trend structure and volume spikes.
# Long when: weekly uptrend (higher weekly high & higher weekly low), volume > 1.5x 20-day average, and price reverses from weekly S1 support.
# Short when: weekly downtrend (lower weekly high & lower weekly low), volume > 1.5x 20-day average, and price reverses from weekly R1 resistance.
# Exit when price reverses from opposite pivot level or weekly trend breaks.
# Designed to capture reversals at key weekly levels in both bull and bear markets, with volume confirmation to avoid false signals.
# Weekly pivot provides structure, reversal captures mean reversion at extremes, weekly trend filter ensures directional bias.

name = "1d_WeeklyPivot_Reversal_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot points and trend structure
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly pivot point calculation (based on prior week) ---
    # Pivot point, support/resistance levels for current week
    # Using prior week's OHLC to calculate current week's pivots
    high_wk = df_weekly['high'].values
    low_wk = df_weekly['low'].values
    close_wk = df_weekly['close'].values
    
    # Calculate pivot points for each week (based on prior week data)
    pp = np.zeros_like(high_wk)
    r1 = np.zeros_like(high_wk)
    s1 = np.zeros_like(high_wk)
    r2 = np.zeros_like(high_wk)
    s2 = np.zeros_like(high_wk)
    
    for i in range(1, len(high_wk)):  # Start from 1 to use prior week data
        # Prior week's OHLC
        ph = high_wk[i-1]
        pl = low_wk[i-1]
        pc = close_wk[i-1]
        
        # Standard pivot point calculation
        pp[i] = (ph + pl + pc) / 3.0
        r1[i] = 2 * pp[i] - pl
        s1[i] = 2 * pp[i] - ph
        r2[i] = pp[i] + (ph - pl)
        s2[i] = pp[i] - (ph - pl)
    
    # First week: no prior week, set to NaN
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    r2[0] = np.nan
    s2[0] = np.nan
    
    # --- Weekly trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    # Higher High: this week's high > prior week's high
    hh = high_wk[1:] > high_wk[:-1]
    # Higher Low: this week's low > prior week's low
    hl = low_wk[1:] > low_wk[:-1]
    # Lower High: this week's high < prior week's high
    lh = high_wk[1:] < high_wk[:-1]
    # Lower Low: this week's low < prior week's low
    ll = low_wk[1:] < low_wk[:-1]
    
    # Build trend arrays (same length as weekly data)
    uptrend_wk = np.zeros_like(high_wk, dtype=bool)
    downtrend_wk = np.zeros_like(high_wk, dtype=bool)
    
    # Skip first week (no prior week for comparison)
    uptrend_wk[1:] = hh & hl
    downtrend_wk[1:] = lh & ll
    # First week: no trend
    uptrend_wk[0] = False
    downtrend_wk[0] = False
    
    # --- Volume confirmation (volume > 20-day average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly data to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    uptrend_aligned = align_htf_to_ltf(prices, df_weekly, uptrend_wk)
    downtrend_aligned = align_htf_to_ltf(prices, df_weekly, downtrend_wk)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly data (at least 2 weeks) and volume MA(20)
    start_idx = max(20, 1)  # volume MA needs 20, weekly data needs at least index 1
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend from aligned data
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: weekly uptrend + volume spike + price at or below S1 support (reversal long)
                if close[i] <= s1_aligned[i] * 1.001:  # Allow small buffer for slippage
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: weekly downtrend + volume spike + price at or above R1 resistance (reversal short)
                if close[i] >= r1_aligned[i] * 0.999:  # Allow small buffer for slippage
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price reaches or exceeds R1 resistance OR weekly uptrend breaks
                if close[i] >= r1_aligned[i] * 0.999 or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches or falls below S1 support OR weekly downtrend breaks
                if close[i] <= s1_aligned[i] * 1.001 or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals