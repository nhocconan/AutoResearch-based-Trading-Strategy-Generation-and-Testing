#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Uses Camarilla pivot levels from 1d timeframe for breakout entries on 12h, filtered by daily trend structure and volume spikes.
# Long when: daily uptrend (HH & HL), volume > 1.5x 20-period average, and price breaks above R1 (bullish breakout).
# Short when: daily downtrend (LH & LL), volume > 1.5x 20-period average, and price breaks below S1 (bearish breakdown).
# Exit when price returns to the pivot point (mean reversion) or daily trend breaks.
# Camarilla levels provide high-probability reversal zones; breakouts from these levels with trend and volume filters capture strong moves while avoiding false signals.
# Works in bull markets by catching upside breakouts in uptrends and in bear markets by catching downside breakdowns in downtrends.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and trend structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla pivot levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    # Camarilla levels
    r1 = pivot + range_1d * 1.1 / 12
    s1 = pivot - range_1d * 1.1 / 12
    
    # --- 1d trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    hh = high_1d > np.roll(high_1d, 1)
    hl = low_1d > np.roll(low_1d, 1)
    lh = high_1d < np.roll(high_1d, 1)
    ll = low_1d < np.roll(low_1d, 1)
    uptrend = hh & hl
    downtrend = lh & ll
    uptrend[0] = False
    downtrend[0] = False
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (needs 2 days) and volume MA(20)
    start_idx = max(2, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: daily uptrend + volume spike + price breaks above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + price breaks below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to pivot OR daily uptrend breaks
                if close[i] <= pivot_aligned[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot OR daily downtrend breaks
                if close[i] >= pivot_aligned[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals