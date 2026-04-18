#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout
Hypothesis: Use weekly pivot points (S1, S2, R1, R2) as dynamic support/resistance levels.
Go long when price breaks above R1 with volume confirmation and weekly trend up (price > weekly close).
Go short when price breaks below S1 with volume confirmation and weekly trend down (price < weekly close).
Weekly pivot provides strong institutional levels that work in both bull and bear markets.
Targets 15-25 trades/year by requiring pivot break, volume > 1.5x 20-day average, and weekly trend alignment.
Works in bull markets by buying breaks above weekly resistance, and in bear markets by selling breaks below weekly support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend (HTF)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    
    # Align weekly levels to daily timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Weekly trend: price above/below weekly close
    weekly_close_aligned = align_htf_to_ltf(prices, df_weekly, close_weekly)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_close_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1, with volume, and weekly trend up (close > weekly close)
            if (close[i] > r1_aligned[i] and vol_confirm[i] and 
                close[i] > weekly_close_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1, with volume, and weekly trend down (close < weekly close)
            elif (close[i] < s1_aligned[i] and vol_confirm[i] and 
                  close[i] < weekly_close_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below weekly close (trend change) or breaks below S1
            if (close[i] < weekly_close_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly close (trend change) or breaks above R1
            if (close[i] > weekly_close_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_Breakout"
timeframe = "1d"
leverage = 1.0