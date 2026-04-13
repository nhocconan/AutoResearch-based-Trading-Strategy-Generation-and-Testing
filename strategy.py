#!/usr/bin/env python3
"""
4h_1d_Weekly_Trend_Breakout_v2
Hypothesis: Trade breakouts from weekly high/low on 4h timeframe with volume confirmation and 1d trend filter.
Weekly high/low act as strong support/resistance in both bull and bear markets. 
Breakouts above weekly high signal institutional buying; breakdowns below weekly low signal distribution.
1d EMA50 ensures trades align with intermediate-term trend. Volume filter confirms participation.
Target: 20-30 trades/year to minimize fee drift.
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
    
    # Get weekly data for high/low
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Align weekly high/low to 4h
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, high_weekly)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, low_weekly)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.5x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above weekly high with volume expansion and price above daily EMA50
        long_condition = (close[i] > weekly_high_aligned[i]) and volume_expansion[i] and (close[i] > ema_50_aligned[i])
        
        # Short: breakdown below weekly low with volume expansion and price below daily EMA50
        short_condition = (close[i] < weekly_low_aligned[i]) and volume_expansion[i] and (close[i] < ema_50_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Weekly_Trend_Breakout_v2"
timeframe = "4h"
leverage = 1.0