#!/usr/bin/env python3
"""
12h_Pivot_Reversal_1dTrend_Filter
Hypothesis: On 12-hour timeframe, enter long at daily pivot support with bullish 1d trend (close > EMA50) and volume confirmation, short at daily pivot resistance with bearish 1d trend (close < EMA50) and volume confirmation. Exit on opposite pivot level break. Uses daily pivot points (based on prior day's OHLC) for precise reversal entries in ranging markets, filtered by 1d trend to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) to minimize fee decay in both bull and bear markets.
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
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Trend: bullish when price > EMA50, bearish when price < EMA50
    d1_uptrend = close > ema_50_aligned
    d1_downtrend = close < ema_50_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup (EMA50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 1d EMA50 trend alignment and volume confirmation
        long_entry = close[i] <= s1_aligned[i] and d1_uptrend[i] and volume_surge[i]
        short_entry = close[i] >= r1_aligned[i] and d1_downtrend[i] and volume_surge[i]
        
        # Exit on opposite pivot level break
        long_exit = close[i] >= r1_aligned[i]
        short_exit = close[i] <= s1_aligned[i]
        
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

name = "12h_Pivot_Reversal_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0