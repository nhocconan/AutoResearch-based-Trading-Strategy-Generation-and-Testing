#!/usr/bin/env python3
# 1d_Weekly_Pivot_Breakout_Momentum
# Hypothesis: Trade breakouts from weekly pivot levels on daily chart with momentum confirmation.
# Uses weekly pivot points (R1, S1) as key support/resistance levels.
# Long when price breaks above R1 with bullish momentum (MACD > 0), short when breaks below S1 with bearish momentum (MACD < 0).
# Weekly context provides structural levels that work in both bull and bear markets.
# Targets ~15-25 trades/year to minimize fee drag on daily timeframe.

name = "1d_Weekly_Pivot_Breakout_Momentum"
timeframe = "1d"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily MACD for momentum confirmation (12,26,9)
    exp1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    exp2 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean()
    macd = exp1 - exp2
    signal_line = pd.Series(macd).ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for MACD
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(macd_hist[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with bullish momentum
            if close[i] > r1_aligned[i] and macd_hist[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with bearish momentum
            elif close[i] < s1_aligned[i] and macd_hist[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: break below pivot or momentum turns bearish
            if close[i] < pivot_aligned[i] or macd_hist[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: break above pivot or momentum turns bullish
            if close[i] > pivot_aligned[i] or macd_hist[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals