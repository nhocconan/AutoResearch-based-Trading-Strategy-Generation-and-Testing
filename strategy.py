#!/usr/bin/env python3
"""
6h_1d_1w_Power_Trend_Aligned
Hypothesis: Combines daily Elder Ray (bull/bear power) with weekly trend alignment and volume confirmation on 6h.
Elder Ray measures bull power (high - EMA13) and bear power (EMA13 - low) to detect institutional buying/selling pressure.
Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
Volume confirms institutional participation.
Works in bull markets via strong bull power + uptrend, and in bear markets via strong bear power + downtrend.
Target: 20-40 trades/year on 6h (80-160 total over 4 years).
"""

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
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA13 on daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_1d - ema13_1d  # Bull power: high minus EMA
    bear_power = ema13_1d - low_1d   # Bear power: EMA minus low
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = close_1w > ema21_1w
    weekly_downtrend = close_1w < ema21_1w
    
    # Align all signals to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Volume confirmation on 6h: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirmation = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: strong bull power + weekly uptrend + volume confirmation
        long_condition = (bull_power_aligned[i] > 0) and weekly_uptrend_aligned[i] and volume_confirmation[i]
        
        # Short conditions: strong bear power + weekly downtrend + volume confirmation
        short_condition = (bear_power_aligned[i] > 0) and weekly_downtrend_aligned[i] and volume_confirmation[i]
        
        # Exit conditions: power weakening or trend reversal
        exit_long = (bull_power_aligned[i] <= 0) or not weekly_uptrend_aligned[i]
        exit_short = (bear_power_aligned[i] <= 0) or not weekly_downtrend_aligned[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_Power_Trend_Aligned"
timeframe = "6h"
leverage = 1.0