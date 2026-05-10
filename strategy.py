#!/usr/bin/env python3
"""
6h_Weekly_Range_Breakout_Pullback
Hypothesis: Uses weekly range (Monday open to Friday close) to define range boundaries.
Breakouts above weekly high or below weekly low with volume trigger entries.
Pullback to 50% of weekly range with trend continuation allows re-entry.
Designed for 6h timeframe to capture multi-day moves with low frequency.
Works in both bull and bear markets by capturing breakouts and mean-reversion within weekly ranges.
Target: 15-35 trades/year per symbol.
"""

name = "6h_Weekly_Range_Breakout_Pullback"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly range: Monday open to Friday close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_range = weekly_high - weekly_low
    weekly_mid = weekly_low + weekly_range * 0.5
    
    # Use previous week's levels (shift by 1 to avoid look-ahead)
    weekly_high_prev = weekly_high.shift(1).values
    weekly_low_prev = weekly_low.shift(1).values
    weekly_mid_prev = weekly_mid.shift(1).values
    
    # Weekly trend: price vs weekly mid
    weekly_close = df_1w['close'].values
    weekly_trend_up = weekly_close > weekly_mid_prev
    weekly_trend_down = weekly_close < weekly_mid_prev
    
    # Align weekly data to 6h
    weekly_high_prev_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_prev)
    weekly_low_prev_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_prev)
    weekly_mid_prev_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid_prev)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Volume confirmation: 20-period (~5.3 days) average on 6h
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_prev_aligned[i]) or np.isnan(weekly_low_prev_aligned[i]) or
            np.isnan(weekly_mid_prev_aligned[i]) or np.isnan(weekly_trend_up_aligned[i]) or
            np.isnan(weekly_trend_down_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: break above weekly high with weekly uptrend and volume
            if (close[i] > weekly_high_prev_aligned[i] and 
                weekly_trend_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly low with weekly downtrend and volume
            elif (close[i] < weekly_low_prev_aligned[i] and 
                  weekly_trend_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
            # Long pullback: price pulls back to weekly mid in uptrend with volume
            elif (close[i] >= weekly_mid_prev_aligned[i] * 0.97 and 
                  close[i] <= weekly_mid_prev_aligned[i] * 1.03 and
                  weekly_trend_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short pullback: price pulls back to weekly mid in downtrend with volume
            elif (close[i] >= weekly_mid_prev_aligned[i] * 0.97 and 
                  close[i] <= weekly_mid_prev_aligned[i] * 1.03 and
                  weekly_trend_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to weekly low or trend fails
            if (close[i] < weekly_low_prev_aligned[i] or 
                weekly_trend_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to weekly high or trend fails
            if (close[i] > weekly_high_prev_aligned[i] or 
                weekly_trend_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals