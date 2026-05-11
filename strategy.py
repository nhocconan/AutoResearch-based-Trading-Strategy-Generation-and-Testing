#!/usr/bin/env python3
"""
4h_SuperTrend_Breakout_v1
Hypothesis: In trending markets, price breaking above/below the SuperTrend with volume confirmation provides high-probability entries.
In ranging markets, the Choppiness Index filter prevents false signals. Works in both bull and bear regimes by adapting to trend strength.
Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_SuperTrend_Breakout_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1D Data for Choppiness Index (Regime Filter) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR (14)
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Sum of True Range over 14 periods
    tr_sum = np.zeros_like(tr)
    for i in range(13, len(tr)):
        tr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(tr)
    for i in range(13, len(tr)):
        if tr_sum[i] > 0 and atr_1d[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (atr_1d[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50.0
    
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # === SuperTrend Calculation (4h) ===
    atr_period = 10
    multiplier = 3.0
    
    # True Range for SuperTrend
    tr1_st = np.abs(high - low)
    tr2_st = np.abs(high - np.roll(close, 1))
    tr3_st = np.abs(low - np.roll(close, 1))
    tr_st = np.maximum(tr1_st, np.maximum(tr2_st, tr3_st))
    tr_st[0] = tr1_st[0]
    
    # ATR
    atr_st = np.zeros_like(tr_st)
    atr_st[0] = tr_st[0]
    for i in range(1, len(tr_st)):
        atr_st[i] = (atr_st[i-1] * (atr_period-1) + tr_st[i]) / atr_period
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr_st
    basic_lb = (high + low) / 2 - multiplier * atr_st
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close)):
        # Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # SuperTrend
    super_trend = np.zeros_like(close)
    trend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    super_trend[0] = final_lb[0]
    trend[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > final_ub[i-1]:
            super_trend[i] = final_lb[i]
            trend[i] = 1
        elif close[i] < final_lb[i-1]:
            super_trend[i] = final_ub[i]
            trend[i] = -1
        else:
            super_trend[i] = super_trend[i-1]
            trend[i] = trend[i-1]
            if trend[i] == 1 and final_lb[i] < super_trend[i-1]:
                super_trend[i] = final_lb[i]
            if trend[i] == -1 and final_ub[i] > super_trend[i-1]:
                super_trend[i] = final_ub[i]
    
    # === Volume Filter (4h) ===
    vol_ma = np.zeros_like(volume)
    vol_ma[0] = volume[0]
    for i in range(1, len(volume)):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20  # 20-period MA
    
    volume_ratio = volume / (vol_ma + 1e-10)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_align[i]) or 
            np.isnan(super_trend[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above SuperTrend in uptrend with volume confirmation and not choppy
            if (close[i] > super_trend[i] and 
                close[i-1] <= super_trend[i-1] and  # crossed above
                trend[i] == 1 and 
                volume_ratio[i] > 1.5 and 
                chop_align[i] < 61.8):  # not choppy (trending)
                signals[i] = 0.25
                position = 1
            # Short: price crosses below SuperTrend in downtrend with volume confirmation and not choppy
            elif (close[i] < super_trend[i] and 
                  close[i-1] >= super_trend[i-1] and  # crossed below
                  trend[i] == -1 and 
                  volume_ratio[i] > 1.5 and 
                  chop_align[i] < 61.8):  # not choppy (trending)
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below SuperTrend
            if close[i] < super_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above SuperTrend
            if close[i] > super_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals