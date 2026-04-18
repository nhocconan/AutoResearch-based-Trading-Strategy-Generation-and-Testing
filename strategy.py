#!/usr/bin/env python3
"""
4h_1D_WeeklyPivot_PivotReversal_Volume
Hypothesis: Uses weekly pivot levels as structural support/resistance with volume confirmation
and 1d EMA trend filter. Mean-reverts at weekly pivot levels in ranging markets while
following trend in trending conditions. Targets 25-35 trades/year.
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: (H+L+C)/3
    typical_price = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3
    pivot = typical_price.values
    # Weekly R1: 2*P - L
    r1 = 2 * pivot - df_weekly['low'].values
    # Weekly S1: 2*P - H
    s1 = 2 * pivot - df_weekly['high'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align weekly pivots to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price near S1 with volume spike and 1d uptrend
            if (close[i] <= s1_aligned[i] * 1.005 and vol_spike[i] and 
                close[i] > ema_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price near R1 with volume spike and 1d downtrend
            elif (close[i] >= r1_aligned[i] * 0.995 and vol_spike[i] and 
                  close[i] < ema_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or trend turns down
            if (close[i] >= pivot_aligned[i] * 0.995 or close[i] < ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or trend turns up
            if (close[i] <= pivot_aligned[i] * 1.005 or close[i] > ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_WeeklyPivot_PivotReversal_Volume"
timeframe = "4h"
leverage = 1.0