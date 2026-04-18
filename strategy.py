# SPDX-FileCopyrightText: 2025 Agent42
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""
12h Pivot Range Breakout with Volume and Weekly Trend Filter
Hypothesis: Price breaks of daily pivot ranges (R1/S1) on 12h timeframe with volume
confirmation and weekly trend filter capture institutional breakout moves in both
bull and bear markets. The weekly EMA34 ensures we trade with dominant higher timeframe
momentum, reducing counter-trend trades. Pivot ranges provide clear support/resistance
levels that work across regimes. Volume confirmation filters false breakouts.
Target: 15-25 trades/year to minimize fee drag while capturing strong momentum.
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
    
    # Get 1d data for pivot points (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate pivot points from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align pivot levels to 12h timeframe (they update daily)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.8x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        trend = ema34_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume, in uptrend
            if price > r1_val and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume, in downtrend
            elif price < s1_val and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to pivot or trend weakens
            if price < pivot_aligned[i] or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to pivot or trend weakens
            if price > pivot_aligned[i] or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_Range_Breakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0