#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_Pivot_Breakout_Trend
Hypothesis: Combines weekly trend bias from weekly close vs weekly open with Camarilla pivot breakouts on daily timeframe.
In bullish weeks (weekly close > weekly open), we look for long breakouts above R3 on daily; in bearish weeks (weekly close < weekly open), we look for short breakdowns below S3.
Uses volume confirmation to avoid false breakouts. Works in both bull and bear markets by adapting to weekly trend.
Target: 12-37 trades/year on 6h (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using typical price: (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R3 = typical_price + 1.1 * range_val * 1.1 / 2
    S3 = typical_price - 1.1 * range_val * 1.1 / 2
    R4 = typical_price + 1.1 * range_val * 1.5
    S4 = typical_price - 1.1 * range_val * 1.5
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_ltf_to_htf(prices, df_1w, weekly_bullish)
    
    # Align daily Camarilla levels to 6h timeframe
    R3_aligned = align_ltf_to_htf(prices, df_1d, R3.values)
    S3_aligned = align_ltf_to_htf(prices, df_1d, S3.values)
    R4_aligned = align_ltf_to_htf(prices, df_1d, R4.values)
    S4_aligned = align_ltf_to_htf(prices, df_1d, S4.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias
        is_bullish_week = weekly_bullish_aligned[i]
        
        # Entry conditions
        long_signal = False
        short_signal = False
        
        if is_bullish_week:
            # Bullish week: look for long breakouts above R3 with volume
            if close[i] > R3_aligned[i] and volume_expansion[i]:
                long_signal = True
        else:
            # Bearish week: look for short breakdowns below S3 with volume
            if close[i] < S3_aligned[i] and volume_expansion[i]:
                short_signal = True
        
        # Exit conditions: reverse signal or loss of volume expansion
        if position == 1 and (short_signal or not volume_expansion[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_signal or not volume_expansion[i]):
            position = 0
            signals[i] = 0.0
        elif position == 0 and long_signal:
            position = 1
            signals[i] = position_size
        elif position == 0 and short_signal:
            position = -1
            signals[i] = -position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_Camarilla_Pivot_Breakout_Trend"
timeframe = "6h"
leverage = 1.0