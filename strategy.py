#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_1dTrend_Filter
Hypothesis: Uses weekly pivot points (from weekly high/low/close) as support/resistance, combined with Donchian(20) breakout on 6h and 1d EMA200 trend filter to capture institutional breakouts. Weekly pivots provide stronger S/R than daily; Donchian captures breakouts; 1d EMA200 filters trend direction. Designed for 25-40 trades/year (~100-160 total over 4 years) to minimize fee drift while capturing strong directional moves in both bull and bear markets.
"""

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
    
    # Get weekly data for pivot points (HLC/3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: (H+L+C)/3; R1 = 2*P - L; S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian(20) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 48-period MA (8 days of 6h bars)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Volume confirmation (>1.8x average)
        vol_confirm = volume[i] > (1.8 * vol_ma_48[i])
        
        # Breakout conditions at weekly S1/R1 with Donchian confirmation
        long_breakout = (close[i] > r1_aligned[i] and 
                         close[i] > donchian_high[i] and 
                         vol_confirm and 
                         uptrend)
        short_breakout = (close[i] < s1_aligned[i] and 
                          close[i] < donchian_low[i] and 
                          vol_confirm and 
                          downtrend)
        
        # Exit conditions: return to weekly pivot
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0