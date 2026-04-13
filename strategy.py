#!/usr/bin/env python3
"""
6h_1d_Liquidity_Zone_Trend
Hypothesis: Combines liquidity zones from daily timeframe (equal highs/lows) with 6h trend to capture breakouts.
Liquidity zones act as support/resistance where stops accumulate. Breakouts from these zones with volume and 6h trend alignment
provide high-probability trades. Works in both bull and bear markets by following the 6h trend direction.
Target: 15-30 trades/year on 6h (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def find_liquidity_zones(high, low, window=20, tolerance=0.001):
    """Find liquidity zones (equal highs/lows) within tolerance."""
    n = len(high)
    liquidity_high = np.full(n, np.nan)
    liquidity_low = np.full(n, np.nan)
    
    for i in range(window, n):
        # Check for equal highs
        high_window = high[i-window:i]
        max_high = np.max(high_window)
        if np.sum(np.abs(high_window - max_high) <= tolerance * max_high) >= 2:
            liquidity_high[i] = max_high
        
        # Check for equal lows
        low_window = low[i-window:i]
        min_low = np.min(low_window)
        if np.sum(np.abs(low_window - min_low) <= tolerance * min_low) >= 2:
            liquidity_low[i] = min_low
    
    return liquidity_high, liquidity_low

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for liquidity zones
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate liquidity zones on daily
    liq_high_1d, liq_low_1d = find_liquidity_zones(high_1d, low_1d, window=20, tolerance=0.0005)
    
    # Align liquidity zones to 6h timeframe
    liq_high_1d_aligned = align_htf_to_ltf(prices, df_1d, liq_high_1d)
    liq_low_1d_aligned = align_htf_to_ltf(prices, df_1d, liq_low_1d)
    
    # 6h trend: EMA(21) vs EMA(50)
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = ema_21 > ema_50
    trend_down = ema_21 < ema_50
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(liq_high_1d_aligned[i]) or np.isnan(liq_low_1d_aligned[i]) or
            np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Breakout above liquidity high with uptrend and volume expansion
        long_breakout = (high[i] > liq_high_1d_aligned[i]) and trend_up[i] and volume_expansion[i]
        
        # Breakdown below liquidity low with downtrend and volume expansion
        short_breakdown = (low[i] < liq_low_1d_aligned[i]) and trend_down[i] and volume_expansion[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif long_breakout and position == 1:
            signals[i] = position_size
        elif short_breakdown and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_breakdown and position == -1:
            signals[i] = -position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_Liquidity_Zone_Trend"
timeframe = "6h"
leverage = 1.0