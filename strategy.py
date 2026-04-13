#!/usr/bin/env python3
"""
12h_1d_Weekly_Range_Breakout_With_Volume_Confirmation
Hypothesis: Weekly price ranges provide strong structural support/resistance.
Breakouts above weekly high or below weekly low with volume confirmation
capture institutional participation. Trend filter uses 1d EMA50 to align with
medium-term direction. Works in bull (continuation) and bear (mean reversion at
extremes) by requiring volume and trend alignment. Targets 12-37 trades/year.
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
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly high/low to 12h
    weekly_high = align_htf_to_ltf(prices, df_1w, high_1w)
    weekly_low = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above weekly high with volume expansion
        # 2. Must be above 1d EMA50 for trend alignment
        breakout_long = (close[i] > weekly_high[i]) and volume_expansion[i]
        long_condition = breakout_long and (close[i] > ema_50_aligned[i])
        
        # Short conditions:
        # 1. Breakdown below weekly low with volume expansion
        # 2. Must be below 1d EMA50 for trend alignment
        breakdown_short = (close[i] < weekly_low[i]) and volume_expansion[i]
        short_condition = breakdown_short and (close[i] < ema_50_aligned[i])
        
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

name = "12h_1d_Weekly_Range_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0