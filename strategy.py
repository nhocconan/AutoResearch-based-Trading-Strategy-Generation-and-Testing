#!/usr/bin/env python3
"""
6H_WeeklyPivot_DailyTrend_VolumeBreakout
Hypothesis: Weekly pivots (from 1w) define institutional support/resistance, while daily trend (from 1d) filters direction, and volume breakout confirms institutional participation. Works in bull markets by buying dips to weekly support in uptrends, and in bear markets by selling rallies to weekly resistance in downtrends. Uses 6h timeframe for balance of signal frequency and cost efficiency.
"""

name = "6H_WeeklyPivot_DailyTrend_VolumeBreakout"
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support 1: S1 = 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    # Resistance 1: R1 = 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = np.zeros_like(close_1d)
    ema_34_1d[:] = np.nan
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_34_1d[i-1] * (33 / (34 + 1)))
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Align daily EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    vol_ma_20[:] = np.nan
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price near weekly S1 (within 0.5%), daily uptrend, volume spike
            near_s1 = abs(close[i] - s1_1w_aligned[i]) / s1_1w_aligned[i] < 0.005
            daily_uptrend = close[i] > ema_34_1d_aligned[i]
            
            if near_s1 and daily_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price near weekly R1 (within 0.5%), daily downtrend, volume spike
            elif abs(close[i] - r1_1w_aligned[i]) / r1_1w_aligned[i] < 0.005 and \
                 close[i] < ema_34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot or daily trend reverses
            if close[i] >= pivot_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot or daily trend reverses
            if close[i] <= pivot_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals