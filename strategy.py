#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Daily_Trend_Continuation
Hypothesis: Weekly pivot levels from 1w provide strong structural support/resistance.
Continuations above weekly R4 or below weekly S4 are traded only when aligned with 1d EMA34 trend
and confirmed by volume spikes, targeting breakout moves in both bull and bear markets.
Designed for low turnover (~15-25 trades/year) to minimize fee drag on 6h timeframe.
"""

name = "6h_Weekly_Pivot_Daily_Trend_Continuation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Levels (R4/S4 for breakout) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r4 = pivot_1w + (range_1w * 1.1)
    s4 = pivot_1w - (range_1w * 1.1)
    
    # Align weekly levels to 6h
    r4_6h = align_htf_to_ltf(prices, df_1w, r4)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4)
    
    # === 1d EMA34 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (24-period EMA for 6h) ===
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ok = volume > vol_ema24 * 2.0  # Require 2x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for indicators)
    start_idx = 100  # covers EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema34_1d_6h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above weekly R4 + above 1d EMA34 + volume spike
            if close[i] > r4_6h[i] and close[i] > ema34_1d_6h[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Break below weekly S4 + below 1d EMA34 + volume spike
            elif close[i] < s4_6h[i] and close[i] < ema34_1d_6h[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: reverse signal or trend failure
            if position == 1:
                # Exit: Price breaks below weekly S4 OR closes below 1d EMA34
                if close[i] < s4_6h[i] or close[i] < ema34_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price breaks above weekly R4 OR closes above 1d EMA34
                if close[i] > r4_6h[i] or close[i] > ema34_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals