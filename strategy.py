#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout
Hypothesis: Use weekly pivot points (from 1w HTF) to determine directional bias, 
then trade Donchian channel breakouts (20-period) on 6h chart only when aligned 
with weekly pivot bias. Volume confirmation filters out weak breakouts.
Works in both bull and bear markets by following weekly pivot trend and using 
volatility-adjusted breakouts with volume filter. Targets 15-25 trades/year per 
symbol with discrete position sizing (0.25) to minimize fee drag.
"""

name = "6h_WeeklyPivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Donchian channel (20-period) - highest high and lowest low
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate ATR(14) for volume filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Get weekly data for pivot points (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_pivot = np.full(len(weekly_close), np.nan)
    weekly_r1 = np.full(len(weekly_close), np.nan)
    weekly_s1 = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            weekly_pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            weekly_r1[i] = 2 * weekly_pivot[i] - weekly_low[i]
            weekly_s1[i] = 2 * weekly_pivot[i] - weekly_high[i]
    
    # Align weekly pivot points to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Ensure Donchian and volume filters are ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(weekly_pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly bias: above pivot = bullish bias, below pivot = bearish bias
        weekly_bias = 1 if close[i] > weekly_pivot_aligned[i] else -1
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above Donchian high with bullish weekly bias and volume confirmation
            if close[i] > donchian_high[i] and weekly_bias == 1 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with bearish weekly bias and volume confirmation
            elif close[i] < donchian_low[i] and weekly_bias == -1 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals