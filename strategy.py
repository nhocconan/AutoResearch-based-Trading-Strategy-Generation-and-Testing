#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Pivot_Trend
Hypothesis: Uses weekly trend (EMA21) to filter 12h Camarilla pivot long/short signals.
Only takes longs when price > weekly EMA21 and shorts when price < weekly EMA21.
Enters at 12h close beyond Camarilla H3/L3 levels with volume > 1.5x 20-period average.
Exits when price returns to Camarilla H4/L4 or on opposite signal.
Designed for 12h timeframe to capture multi-day trends with low frequency (target: 15-35 trades/year).
Works in bull/bear via trend filter and volatility-based pivot levels.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on prior day)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    daily_range = high_1d - low_1d
    H4 = close_1d + 1.5 * daily_range
    H3 = close_1d + 1.0 * daily_range
    L3 = close_1d - 1.0 * daily_range
    L4 = close_1d - 1.5 * daily_range
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    weekly_ema21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align all to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_ok = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or \
           np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or \
           np.isnan(weekly_ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check trend condition
        uptrend = close[i] > weekly_ema21_aligned[i]
        downtrend = close[i] < weekly_ema21_aligned[i]
        
        # Long conditions: uptrend + price closes above H3 with volume
        if uptrend and close[i] > H3_aligned[i] and volume_ok[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short conditions: downtrend + price closes below L3 with volume
        elif downtrend and close[i] < L3_aligned[i] and volume_ok[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Exit conditions: price reaches H4/L4 or trend reverses
        elif position == 1 and (close[i] >= H4_aligned[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= L4_aligned[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        # Hold current position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_Camarilla_Pivot_Trend"
timeframe = "12h"
leverage = 1.0