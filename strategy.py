#!/usr/bin/env python3
"""
12h_Volume_Weighted_Pivot_Breakout
Hypothesis: Combines 12-hour price action with daily pivot points and volume confirmation.
In bull markets, breaks above R1 with volume indicate strength. In bear markets, breaks below S1 indicate weakness.
Uses volume-weighted price action to filter false breakouts and maintain low trade frequency.
Works in both bull and bear regimes by following institutional volume-backed moves.
"""

name = "12h_Volume_Weighted_Pivot_Breakout"
timeframe = "12h"
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
    
    # Get daily data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Align pivot levels to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Price action filter: avoid choppy markets
    # Calculate 12-period price range as percentage of price
    price_range = pd.Series(high - low).rolling(window=12, min_periods=12).mean().values
    avg_price = pd.Series(close).rolling(window=12, min_periods=12).mean().values
    range_pct = price_range / avg_price
    # Only trade when volatility is moderate (not too choppy, not too volatile)
    volatility_filter = (range_pct > 0.01) & (range_pct < 0.05)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and decent volatility
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation and decent volatility
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns below pivot point
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above pivot point
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals