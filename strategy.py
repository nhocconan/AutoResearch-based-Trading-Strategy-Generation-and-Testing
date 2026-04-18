#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter
Hypothesis: Elder Ray's Bull Power (close-EMA13) and Bear Power (EMA13-low) with 1d EMA50 trend filter.
Long when Bull Power > 0 and price above 1d EMA50; short when Bear Power < 0 and price below 1d EMA50.
Works in bull (buy strength) and bear (sell weakness) regimes by aligning with higher timeframe trend.
Target: 20-40 trades/year on 6h timeframe with low frequency and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for Elder Ray calculation
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13, Bear Power = EMA13 - Low
    bull_power = close - ema13
    bear_power = ema13 - low
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power positive (bullish momentum) and above 1d EMA50 (uptrend)
            if bull_power[i] > 0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive (bearish momentum) and below 1d EMA50 (downtrend)
            elif bear_power[i] > 0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or price breaks below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns negative or price breaks above 1d EMA50
            if bear_power[i] <= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter"
timeframe = "6h"
leverage = 1.0