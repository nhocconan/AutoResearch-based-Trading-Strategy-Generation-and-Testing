#!/usr/bin/env python3
"""
6h_1D_WeeklyPivot_Donchian20_Breakout_Volume
Hypothesis: Combines weekly pivot point trend filter with daily Donchian breakout and volume confirmation for 6h timeframe.
Uses weekly pivot levels (from prior week) to establish trend direction: price above weekly pivot = long bias, below = short bias.
Enters on breakout of daily Donchian channel (20-period) in direction of weekly trend, confirmed by volume > 1.5x 20-period average.
Exits when price returns to weekly pivot level or Donchian middle band.
Designed to work in both bull (follow weekly uptrend breaks) and bear (short weekly downtrend breaks) markets.
Targets 15-25 trades/year by requiring weekly trend alignment, daily breakout, and volume confirmation.
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
    
    # Get weekly data for pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C)/3, Support1 = (2*Pivot) - H, Resistance1 = (2*Pivot) - L
    pivot_1w = np.full_like(close_1w, np.nan)
    support1_1w = np.full_like(close_1w, np.nan)
    resistance1_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        support1_1w[i] = (2 * pivot_1w[i]) - high_1w[i-1]
        resistance1_1w[i] = (2 * pivot_1w[i]) - low_1w[i-1]
    
    # Align weekly pivot levels to 6h timeframe (wait for week close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    support1_1w_aligned = align_htf_to_ltf(prices, df_1w, support1_1w)
    resistance1_1w_aligned = align_htf_to_ltf(prices, df_1w, resistance1_1w)
    
    # Get daily data for Donchian channel (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian channel (20-period)
    high_20 = np.full_like(high_1d, np.nan)
    low_20 = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            high_20[i] = np.max(high_1d[i-19:i+1])
            low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian channels to 6h timeframe (wait for day close)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # need volume MA and at least one week of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above daily Donchian high, with volume, and price above weekly pivot
            if (close[i] > high_20_aligned[i] and vol_confirm[i] and 
                close[i] > pivot_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below daily Donchian low, with volume, and price below weekly pivot
            elif (close[i] < low_20_aligned[i] and vol_confirm[i] and 
                  close[i] < pivot_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to weekly pivot or below Donchian low
            if (close[i] <= pivot_1w_aligned[i] or 
                close[i] < low_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot or above Donchian high
            if (close[i] >= pivot_1w_aligned[i] or 
                close[i] > high_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_WeeklyPivot_Donchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0