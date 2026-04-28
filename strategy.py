#!/usr/bin/env python3
"""
4h_WeeklyPivot_R4_S4_Breakout_1dTrend_Volume
Hypothesis: 4-hour chart breakouts at weekly-derived R4/S4 pivot levels with 1-day trend filter and volume confirmation. 
Weekly pivots capture major support/resistance levels that work in both bull and bear markets. 
1-day trend filter ensures trades align with intermediate-term direction. 
Volume confirmation filters false breakouts. Targets 20-50 trades/year by requiring confluence of weekly level break, daily trend, and volume surge.
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
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly R4 and S4 levels (more extreme levels for significant breaks)
    weekly_range = prev_week_high - prev_week_low
    R4 = weekly_pivot + weekly_range * 1.1
    S4 = weekly_pivot - weekly_range * 1.1
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 4h
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above R4 + 1d uptrend + volume surge
        long_entry = (close[i] > R4_aligned[i] and 
                     trend_up[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S4 + 1d downtrend + volume surge
        short_entry = (close[i] < S4_aligned[i] and 
                      trend_down[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < S4_aligned[i] and volume_surge[i]
        short_exit = close[i] > R4_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WeeklyPivot_R4_S4_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0