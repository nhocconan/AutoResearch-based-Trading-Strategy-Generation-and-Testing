#!/usr/bin/env python3
"""
1d_1W_Camarilla_R1S1_Breakout_Volume_Sparse_v3
Hypothesis: Use weekly and daily price action with sparse entries - only trade when weekly trend aligns with daily breakout, volume confirms, and price is outside weekly Bollinger Bands. Target 15-25 trades/year to avoid fee drag. Works in bull via weekly uptrend + daily breakouts, in bear via weekly downtrend + daily breakdowns. Bollinger Band filter ensures we only trade significant moves, reducing whipsaw.
"""

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
    
    # Get weekly data for trend and Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly calculations
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's OHLC for calculations
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # Weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_1w_series = pd.Series(close_1w)
    bb_middle = close_1w_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_1w_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + bb_std_dev * bb_std
    bb_lower = bb_middle - bb_std_dev * bb_std
    
    # Weekly trend: price above/below middle band
    weekly_uptrend = close_1w > bb_middle
    weekly_downtrend = close_1w < bb_middle
    
    # Daily calculations for entry
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Daily Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + range_1d * 1.1 / 12
    s1_1d = prev_close_1d - range_1d * 1.1 / 12
    
    # Align all higher timeframe data to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need enough for BB and averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend, price breaks above daily R1, above weekly upper BB, volume confirms
            if (weekly_uptrend_aligned[i] and 
                close[i] > r1_1d_aligned[i] and 
                close[i] > bb_upper_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend, price breaks below daily S1, below weekly lower BB, volume confirms
            elif (weekly_downtrend_aligned[i] and 
                  close[i] < s1_1d_aligned[i] and 
                  close[i] < bb_lower_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below daily R1 or weekly trend changes
            if close[i] < r1_1d_aligned[i] or not weekly_uptrend_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above daily S1 or weekly trend changes
            if close[i] > s1_1d_aligned[i] or not weekly_downtrend_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_R1S1_Breakout_Volume_Sparse_v3"
timeframe = "1d"
leverage = 1.0