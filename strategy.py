#!/usr/bin/env python3
"""
6h_LongTermTrend_PullbackWithVolume
Hypothesis: Price respects long-term trend (1d EMA89) and tends to pull back to the 20-period EMA on 6h before continuing.
Enter long when price pulls back to EMA20 in uptrend (EMA89 rising) with volume confirmation.
Enter short when price rallies to EMA20 in downtrend (EMA89 falling) with volume confirmation.
Uses volume spike (>1.5x 20-period volume average) to confirm momentum resumption.
Targets 80-160 trades over 4 years (20-40/year) to balance opportunity and cost.
Works in bull (buy dips) and bear (sell rallies) by following the higher timeframe trend.
"""

name = "6h_LongTermTrend_PullbackWithVolume"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d EMA89 for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema89_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 89:
        ema89_1d[88] = np.mean(close_1d[:89])
        alpha = 2 / (89 + 1)
        for i in range(89, len(close_1d)):
            ema89_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema89_1d[i-1]
    ema89_1d_prev = np.roll(ema89_1d, 1)
    ema89_1d_prev[0] = np.nan
    ema89_rising = ema89_1d > ema89_1d_prev
    ema89_falling = ema89_1d < ema89_1d_prev
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    ema89_rising_aligned = align_htf_to_ltf(prices, df_1d, ema89_rising)
    ema89_falling_aligned = align_htf_to_ltf(prices, df_1d, ema89_falling)
    
    # 6h EMA20 for pullback target
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    # 6h volume SMA20 for volume confirmation
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(90, 20)  # Need EMA89 and EMA20
    
    for i in range(start_idx, n):
        if np.isnan(ema89_1d_aligned[i]) or np.isnan(ema89_rising_aligned[i]) or \
           np.isnan(ema89_falling_aligned[i]) or np.isnan(ema20[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        if position == 0:
            # Long: Pullback to EMA20 in uptrend with volume confirmation
            if ema89_rising_aligned[i] and close[i] <= ema20[i] * 1.001 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Rally to EMA20 in downtrend with volume confirmation
            elif ema89_falling_aligned[i] and close[i] >= ema20[i] * 0.999 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Trend reversal or extended move beyond 1.5x EMA20 deviation
            if not ema89_rising_aligned[i] or close[i] >= ema20[i] * 1.03:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend reversal or extended move beyond 1.5x EMA20 deviation
            if not ema89_falling_aligned[i] or close[i] <= ema20[i] * 0.97:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals