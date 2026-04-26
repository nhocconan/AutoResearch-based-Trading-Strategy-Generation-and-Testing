#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_EMA_Trend_Filter
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) zero-lag EMA filtered.
Long when Bull Power > 0 and price > zero-lag EMA21. Short when Bear Power > 0 and price < zero-lag EMA21.
Elder Ray measures bull/bear strength relative to trend. Zero-lag EMA reduces lag for timely entries.
Works in bull markets via Bull Power strength and in bear markets via Bear Power strength.
Discrete sizing 0.25 to minimize fees. Target 12-37 trades/year on 6h.
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
    
    # Get 1d data for zero-lag EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate zero-lag EMA21 on 1d close
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    # Zero-lag EMA: 2*EMA - EMA(EMA)
    ema21_ema_1d = pd.Series(ema_21_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    zl_ema_21_1d = 2 * ema_21_1d - ema21_ema_1d
    zl_ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, zl_ema_21_1d)
    
    # Calculate Elder Ray on 6h: need EMA13
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 21 for EMA)
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(zl_ema_21_1d_aligned[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and price above zero-lag EMA21
            if (bull_power[i] > 0 and close[i] > zl_ema_21_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 and price below zero-lag EMA21
            elif (bear_power[i] > 0 and close[i] < zl_ema_21_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR price crosses below zero-lag EMA21
            if (bull_power[i] <= 0 or close[i] < zl_ema_21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power <= 0 OR price crosses above zero-lag EMA21
            if (bear_power[i] <= 0 or close[i] > zl_ema_21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroLag_EMA_Trend_Filter"
timeframe = "6h"
leverage = 1.0