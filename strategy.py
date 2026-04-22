#!/usr/bin/env python3
"""
Hypothesis: 6-hour ADX + 1-week Williams %R mean reversion.
Long when ADX(14) > 25 (trending) and weekly Williams %R < -80 (oversold).
Short when ADX(14) > 25 (trending) and weekly Williams %R > -20 (overbought).
Exit when ADX drops below 20 (weak trend) or Williams %R crosses centerline (-50).
Uses weekly momentum extremes within strong 6-hour trends to capture mean reversion moves.
Works in both bull and bear markets by filtering for trending conditions (ADX) while using
weekly Williams %R for entry timing in overbought/oversold conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX on 6h data
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Load 1-week data for Williams %R - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Williams %R calculation
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(adx[i]) or np.isnan(williams_r_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong trend + oversold weekly
            if adx[i] > 25 and williams_r_aligned[i] < -80:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend + overbought weekly
            elif adx[i] > 25 and williams_r_aligned[i] > -20:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Weak trend or Williams %R crosses above -50
                if adx[i] < 20 or williams_r_aligned[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Weak trend or Williams %R crosses below -50
                if adx[i] < 20 or williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ADX_1wWilliamsMR"
timeframe = "6h"
leverage = 1.0