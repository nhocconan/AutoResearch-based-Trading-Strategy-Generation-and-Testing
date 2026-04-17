#!/usr/bin/env python3
"""
12h_1w_MultiTimeframe_PivotBreakout_V1
Hypothesis: Combine 1w weekly pivots (CPR) with 1d trend filter and volume confirmation on 12h timeframe. 
Buy when price breaks above weekly CPR pivot with 1d EMA200 uptrend and volume > 1.5x average. 
Sell when breaks below weekly CPR with 1d EMA200 downtrend. 
Exit on opposite CPR break or EMA200 trend reversal. 
Designed for low frequency (<15 trades/year) to minimize fee decay and capture major trend shifts.
"""

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
    
    # === Weekly Data (CPR calculation) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Central Pivot Range (CPR)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    bc = (high_1w + low_1w) / 2.0  # Bottom of CPR
    tc = (pivot * 2) - bc          # Top of CPR
    tc = np.maximum(tc, bc)        # Ensure TC >= BC
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    bc_aligned = align_htf_to_ltf(prices, df_1w, bc)
    tc_aligned = align_htf_to_ltf(prices, df_1w, tc)
    
    # === Daily Data (Trend Filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Volume Confirmation (12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(bc_aligned[i]) or
            np.isnan(tc_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from daily EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above TC (top of CPR) with uptrend and volume
            if close[i] > tc_aligned[i] and uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below BC (bottom of CPR) with downtrend and volume
            elif close[i] < bc_aligned[i] and downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit: price below BC (break CPR down) OR trend reversal
            if close[i] < bc_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above TC (break CPR up) OR trend reversal
            if close[i] > tc_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_MultiTimeframe_PivotBreakout_V1"
timeframe = "12h"
leverage = 1.0