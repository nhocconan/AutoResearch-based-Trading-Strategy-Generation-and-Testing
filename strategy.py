#!/usr/bin/env python3
"""
6h_ElderRay_BullPower_BearPower_1wTrend_v1
Hypothesis: Use Elder Ray's Bull/Bear Power from 1-day data combined with weekly trend filter (EMA200) for 6h entries.
Long when Bull Power > 0 and weekly trend up; short when Bear Power < 0 and weekly trend down.
This captures institutional buying/selling pressure aligned with higher timeframe trend, working in both bull and bear markets.
Target: 20-40 trades per year on 6h timeframe.
"""

name = "6h_ElderRay_BullPower_BearPower_1wTrend_v1"
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
    
    # === 1D Data for Elder Ray (EMA13) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align 1D indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === Weekly Trend Filter (EMA200) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_200_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND weekly uptrend (close > EMA200)
            if bull_power_aligned[i] > 0 and close[i] > ema_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) AND weekly downtrend (close < EMA200)
            elif bear_power_aligned[i] < 0 and close[i] < ema_200_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR trend breaks down
            if bull_power_aligned[i] <= 0 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Bear Power turns positive OR trend breaks up
            if bear_power_aligned[i] >= 0 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals