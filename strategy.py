#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_1dTrend
# Hypothesis: Trade breakouts from weekly pivot levels (PP, R1, S1) only when aligned with daily trend.
# Weekly pivots provide strong support/resistance; daily trend filter ensures we trade with higher timeframe momentum.
# Works in bull markets via breakouts above R1 in uptrend, in bear via breakdowns below S1 in downtrend.
# Uses volume confirmation to avoid false breakouts. Target: 15-25 trades/year.

name = "6h_Weekly_Pivot_Breakout_1dTrend"
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
    
    # Calculate weekly pivot points (using previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high, low, close from previous complete week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Align weekly pivot levels to 6h
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily trend filter: EMA(34) on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.5 * average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above weekly R1 AND price above daily EMA34 (uptrend) AND volume surge
            if close[i] > r1_aligned[i] and close[i] > ema_34_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below weekly S1 AND price below daily EMA34 (downtrend) AND volume surge
            elif close[i] < s1_aligned[i] and close[i] < ema_34_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below weekly PP OR trend bias lost (price below EMA34)
            if close[i] < pp_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above weekly PP OR trend bias lost (price above EMA34)
            if close[i] > pp_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals