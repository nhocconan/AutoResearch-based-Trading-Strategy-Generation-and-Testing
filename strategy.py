# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12H_Camarilla_W1Pivot_Breakout_Trend_Volume
Hypothesis: Weekly pivot levels (R3/S1) combined with 1d EMA trend and volume confirmation on 12h timeframe.
Uses weekly pivot points for structural support/resistance, filtered by daily trend and volume spikes.
Designed to work in both bull and bear markets by following trend direction with strict entry filters.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

name = "12H_Camarilla_W1Pivot_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points: using prior week's high/low/close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Align weekly pivot levels to 12h timeframe
    r3_1w = align_htf_to_ltf(prices, df_1w, r3)
    s1_1w = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: 20-period EMA on 12h data
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # Require 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1w[i]) or np.isnan(s1_1w[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        breakout_long = close[i] > r3_1w[i]
        breakout_short = close[i] < s1_1w[i]
        
        if position == 0:
            # Long: Price breaks above weekly R3 + above daily EMA34 + volume spike
            if breakout_long and price_above_ema1d and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 + below daily EMA34 + volume spike
            elif breakout_short and price_below_ema1d and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse of entry or trend change
            if position == 1:
                # Exit: Price crosses below weekly S1 OR trend turns bearish
                if close[i] < s1_1w[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price crosses above weekly R3 OR trend turns bullish
                if close[i] > r3_1w[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals