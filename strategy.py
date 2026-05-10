#!/usr/bin/env python3
# 6h_ElderRay_Pivot_Reversal
# Hypothesis: Combines Elder Ray (Bull/Bear Power) from 1d with 60-minute Pivot reversals.
# Bull Power > 0 and Bear Power < 0 filter for trend alignment, while price rejection
# at daily pivot levels (R1/S1) provides mean-reversion entries. Works in bull via
# trend-following pullbacks and in bear via oversold/overbought reversals at key levels.
# Low trade frequency expected due to dual-condition requirement.

name = "6h_ElderRay_Pivot_Reversal"
timeframe = "6h"
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
    
    # Get daily data for Elder Ray and Pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # EMA13 for Elder Ray (standard setting)
    ema13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Daily Pivot Points (standard calculation)
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pivot - df_1d['low']
    s1 = 2 * pivot - df_1d['high']
    
    # Align 1d values to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # sufficient warmup for EMA13
    
    for i in range(start_idx, n):
        if np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or \
           np.isnan(bear_power_aligned[i]) or np.isnan(r1_aligned[i]) or \
           np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish bias) and price rejects S1 support
            if bull_power_aligned[i] > 0 and close[i] <= s1_aligned[i] * 1.001 and close[i] > s1_aligned[i] * 0.999:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish bias) and price rejects R1 resistance
            elif bear_power_aligned[i] < 0 and close[i] >= r1_aligned[i] * 0.999 and close[i] <= r1_aligned[i] * 1.001:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or price reaches pivot
            if bull_power_aligned[i] <= 0 or close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive or price reaches pivot
            if bear_power_aligned[i] >= 0 or close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals