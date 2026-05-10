#!/usr/bin/env python3
"""
6h_ElderRay_ForceIndex_Trend_Filter
Hypothesis: Elder Ray (Bull/Bear Power) + Force Index with weekly trend filter captures momentum in both bull and bear markets. 
Elder Ray measures bull/bear power relative to EMA13, Force Index confirms conviction. 
Weekly EMA20 trend filter ensures we trade only in the direction of higher timeframe trend, avoiding counter-trend whipsaws.
Targets 12-30 trades/year (48-120 total) to minimize fee drag.
"""

name = "6h_ElderRay_ForceIndex_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Force Index: (Close - Close_prev) * Volume
    force_index = np.zeros_like(close)
    force_index[1:] = (close[1:] - close[:-1]) * volume[1:]
    # Smooth Force Index with EMA13
    fi_smooth = pd.Series(force_index).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align weekly EMA20 to 6h
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Wait for EMA13 and smoothed FI
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(ema13[i]) or np.isnan(fi_smooth[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend: price above/below weekly EMA20
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND Force Index rising (conviction) AND weekly uptrend
            if bull_power[i] > 0 and fi_smooth[i] > fi_smooth[i-1] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND Force Index falling (conviction) AND weekly downtrend
            elif bear_power[i] < 0 and fi_smooth[i] < fi_smooth[i-1] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative OR Force Index turns negative (loss of conviction)
            if bull_power[i] <= 0 or fi_smooth[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive OR Force Index turns positive (loss of conviction)
            if bear_power[i] >= 0 or fi_smooth[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals