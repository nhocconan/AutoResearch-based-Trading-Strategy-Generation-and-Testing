#!/usr/bin/env python3
"""
6h_1w_1d_Pivot_Breakout_Trend_Follow_v1
Hypothesis: Combines weekly pivot point direction (from 1w) with daily pivot breakout signals (from 1d) and volume confirmation on 6h timeframe.
Trades only in direction of weekly pivot bias (above/below weekly pivot point) to avoid counter-trend trades.
Uses 6h for entry timing with daily pivot breakouts and volume > 1.5x 20-period average.
Designed to work in both bull and bear markets by aligning with higher timeframe trend via weekly pivot.
Target: 20-40 trades/year per symbol.
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
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Daily pivot points: using previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Daily pivot point and support/resistance levels
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot_point - prev_low
    s1 = 2 * pivot_point - prev_high
    r2 = pivot_point + (prev_high - prev_low)
    s2 = pivot_point - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot_point - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot_point)
    
    # Weekly pivot point bias: using weekly OHLC to determine trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        weekly_pivot_bias = np.full(len(prices), np.nan)
    else:
        # Calculate weekly pivot point from weekly data
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        # Align weekly pivot to 6x timeframe (wait for weekly bar to close)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
        # Bias: 1 = bullish (price above weekly pivot), -1 = bearish (price below weekly pivot)
        weekly_pivot_bias = np.where(close > weekly_pivot_aligned, 1, -1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(pivot_point[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(volume_expansion[i]) or np.isnan(weekly_pivot_bias[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above daily R1 with volume expansion and weekly bullish bias
        long_signal = (high[i] > r1[i] and 
                      volume_expansion[i] and 
                      weekly_pivot_bias[i] == 1)
        
        # Short signal: break below daily S1 with volume expansion and weekly bearish bias
        short_signal = (low[i] < s1[i] and 
                       volume_expansion[i] and 
                       weekly_pivot_bias[i] == -1)
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1w_1d_Pivot_Breakout_Trend_Follow_v1"
timeframe = "6h"
leverage = 1.0