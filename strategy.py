#!/usr/bin/env python3
"""
6h_1d_cci_volatility_mean_reversion
Uses daily CCI to identify overbought/oversold conditions and 6h Bollinger Bands for mean reversion entries.
Enters long when daily CCI < -100 (oversold) and price touches lower Bollinger Band on 6h.
Enters short when daily CCI > 100 (overbought) and price touches upper Bollinger Band on 6h.
Exits when price returns to Bollinger Band middle (20-period SMA).
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift.
Works in both bull and bear markets by fading extremes in ranging conditions.
"""

name = "6h_1d_cci_volatility_mean_reversion"
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
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Commodity Channel Index (CCI) on daily timeframe
    # CCI = (Typical Price - SMA(TP, 20)) / (0.015 * Mean Deviation)
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    cci_1d = np.where((mad * 0.015) != 0, (tp_1d - sma_tp) / (mad * 0.015), 0.0)
    
    # Align daily CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Bollinger Bands on 6h timeframe (20-period, 2 standard deviations)
    sma_6h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_6h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_6h + (2 * std_6h)
    lower_bb = sma_6h - (2 * std_6h)
    middle_bb = sma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if CCI data not ready
        if np.isnan(cci_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: daily CCI oversold (< -100) and price touches lower Bollinger Band
        if (cci_aligned[i] < -100 and close[i] <= lower_bb[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: daily CCI overbought (> 100) and price touches upper Bollinger Band
        elif (cci_aligned[i] > 100 and close[i] >= upper_bb[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to middle Bollinger Band
        elif position == 1 and close[i] >= middle_bb[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= middle_bb[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals