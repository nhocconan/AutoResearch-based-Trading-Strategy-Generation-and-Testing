#!/usr/bin/env python3
"""
12h_1w_1d_ema_crossover_volume_v2
Hypothesis: Use 1-week EMA(13) for long-term bias and 1-day EMA(8/21) for medium-term trend alignment, with volume confirmation on 12h breaks. Enter long when 12h close > 1-day EMA(8) with volume > 1.5x average and 1-week EMA rising; short when 12h close < 1-day EMA(21) with volume confirmation and 1-week EMA falling. Designed to capture trends in bull markets and reversals in bear markets by aligning with higher timeframe momentum.
Target: 15-35 trades/year per symbol (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_ema_crossover_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Get weekly data for bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate daily EMA(8) and EMA(21)
    close_1d = df_1d['close'].values
    ema8_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily EMA to 12h timeframe
    ema8_1d_aligned = align_htf_to_ltf(prices, df_1d, ema8_1d)
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Weekly bias using EMA(13) - check if rising/falling
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    # Calculate slope of weekly EMA (rising if current > previous)
    ema13_1w_slope = np.diff(ema13_1w_aligned, prepend=ema13_1w_aligned[0])
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema8_1d_aligned[i]) or np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(ema13_1w_aligned[i]) or np.isnan(ema13_1w_slope[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below daily EMA(21) or weekly EMA turns down
            if close[i] < ema21_1d_aligned[i] or ema13_1w_slope[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above daily EMA(8) or weekly EMA turns up
            if close[i] > ema8_1d_aligned[i] or ema13_1w_slope[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price crosses above daily EMA(8) with volume and weekly EMA rising
            if close[i] > ema8_1d_aligned[i] and vol_confirm[i] and ema13_1w_slope[i] > 0:
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below daily EMA(21) with volume and weekly EMA falling
            elif close[i] < ema21_1d_aligned[i] and vol_confirm[i] and ema13_1w_slope[i] < 0:
                position = -1
                signals[i] = -0.25
    
    return signals