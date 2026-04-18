#!/usr/bin/env python3
"""
12h_MarketStructure_PivotBreakout
12h strategy using daily pivot points (PP) with volume confirmation and weekly trend filter.
- Long: Price crosses above daily pivot + volume > 1.3x daily avg + weekly EMA50 > EMA200
- Short: Price crosses below daily pivot + volume > 1.3x daily avg + weekly EMA50 < EMA200
- Exit: Opposite pivot cross or trend reversal
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (trend continuation) and bear markets (mean reversion at pivot)
"""

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
    
    # Get daily data for pivot points and volume average
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points: PP = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Pivot support/resistance levels
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50w_aligned[i]) or np.isnan(ema_200w_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50w_aligned[i] > ema_200w_aligned[i]
        downtrend = ema_50w_aligned[i] < ema_200w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        # Pivot cross conditions (using close to avoid whipsaw)
        cross_above_pivot = close[i] > pivot_aligned[i] and close[i-1] <= pivot_aligned[i-1]
        cross_below_pivot = close[i] < pivot_aligned[i] and close[i-1] >= pivot_aligned[i-1]
        
        if position == 0:
            # Long: uptrend + volume + cross above daily pivot
            if uptrend and vol_confirm and cross_above_pivot:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + cross below daily pivot
            elif downtrend and vol_confirm and cross_below_pivot:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or cross below pivot
            if not uptrend or (vol_confirm and cross_below_pivot):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or cross above pivot
            if not downtrend or (vol_confirm and cross_above_pivot):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_MarketStructure_PivotBreakout"
timeframe = "12h"
leverage = 1.0