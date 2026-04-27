#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_12hTrend_VolumeConfirmation
Hypothesis: Breakouts from weekly pivot R4/S4 levels with 12h EMA50 trend confirmation and volume spikes (>2x 20-period average) capture momentum in both bull and bear markets. Weekly pivots provide robust support/resistance, reducing false breakouts. Targets 15-30 trades/year to minimize fee drag.
"""

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
    
    # Calculate weekly pivot points (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    
    # Weekly pivot point and support/resistance levels
    pivot_point = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r4 = prev_weekly_close + 3 * (prev_weekly_high - prev_weekly_low)
    weekly_s4 = prev_weekly_close - 3 * (prev_weekly_high - prev_weekly_low)
    
    # Align weekly pivot levels to 6h timeframe (wait for previous week's close)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # 12h EMA50 for trend confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        r4_level = weekly_r4_aligned[i]
        s4_level = weekly_s4_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above weekly R4, above EMA50 trend, volume confirmation
            if close[i] > r4_level and close[i] > ema50_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly S4, below EMA50 trend, volume confirmation
            elif close[i] < s4_level and close[i] < ema50_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA50
            if close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50
            if close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_Breakout_12hTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0