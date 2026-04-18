#!/usr/bin/env python3
"""
6h_RangeBreakout_WeeklyTrend_Volume
Hypothesis: In strong weekly trends (price above/below weekly EMA20), 6h price breaking out of weekly range with volume confirms institutional participation. Works in bull (breakouts above weekly range) and bear (breakdowns below). Volume filter reduces false signals. Targets 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected import name

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.zeros_like(close_1w)
    ema20_1w[:] = np.nan
    for i in range(20, len(close_1w)):
        ema20_1w[i] = np.mean(close_1w[i-20:i])  # Simple MA for stability
    
    # Weekly range (high-low of previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_high = np.zeros_like(close_1w)
    weekly_low = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        weekly_high[i] = high_1w[i-1]
        weekly_low[i] = low_1w[i-1]
    
    # Align weekly indicators to 6h
    ema20_1w_aligned = align_ltf_to_htf(prices, df_1w, ema20_1w)
    weekly_high_aligned = align_ltf_to_htf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_ltf_to_htf(prices, df_1w, weekly_low)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days)
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 24:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-24+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for weekly alignment
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below weekly EMA20
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: weekly uptrend + break above weekly high + volume spike
            if weekly_uptrend and close[i] > weekly_high_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + break below weekly low + volume spike
            elif weekly_downtrend and close[i] < weekly_low_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend fails or price returns below weekly low
            if not weekly_uptrend or close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend fails or price returns above weekly high
            if not weekly_downtrend or close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RangeBreakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0