#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend
Hypothesis: Daily Camarilla pivot levels (R1/S1) act as strong support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and weekly trend alignment 
provide high-probability trades. Weekly trend filter prevents trading against 
major trend, reducing whipsaw in both bull and bear markets. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1 / 12)
    # S1 = C - (Range * 1.1 / 12)
    daily_pivot = (high + low + close) / 3.0
    daily_range = high - low
    r1 = close + (daily_range * 1.1 / 12.0)
    s1 = close - (daily_range * 1.1 / 12.0)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > r1[i-1]  # Break above previous day's R1
        breakout_short = close[i] < s1[i-1]  # Break below previous day's S1
        
        # Trend filter from weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_long and volume_spike[i] and uptrend
        short_entry = breakout_short and volume_spike[i] and downtrend
        
        # Exit on opposite breakout
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
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

name = "1d_Camarilla_Pivot_R1_S1_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0