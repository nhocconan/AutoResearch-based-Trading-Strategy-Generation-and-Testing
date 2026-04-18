#!/usr/bin/env python3
"""
1d_1w_WeeklyPivot_Breakout_RangeFilter_V1
Hypothesis: Breakout of weekly pivot levels (R1/S1) with volume confirmation and 1d trend bias.
Trades only in the direction of the 1d EMA trend to avoid whipsaws in choppy markets.
Targets 7-25 trades per year by using strict weekly pivot levels, volume confirmation, and trend filter.
Works in both bull and bear markets by following the 1d trend.
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
    
    # Get weekly data for pivot levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + (range_1w * 1.1) / 12
    s1_1w = pivot_1w - (range_1w * 1.1) / 12
    
    # Align weekly levels to daily timeframe (wait for bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get 1d trend (EMA34) for directional bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1, above weekly pivot, with volume, and 1d uptrend
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > pivot_1w_aligned[i] and vol_confirm[i] and 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1, below weekly pivot, with volume, and 1d downtrend
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < pivot_1w_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to weekly S1 or 1d downtrend
            if (not np.isnan(s1_1w_aligned[i]) and close[i] < s1_1w_aligned[i]) or \
               (close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly R1 or 1d uptrend
            if (not np.isnan(r1_1w_aligned[i]) and close[i] > r1_1w_aligned[i]) or \
               (close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_WeeklyPivot_Breakout_RangeFilter_V1"
timeframe = "1d"
leverage = 1.0