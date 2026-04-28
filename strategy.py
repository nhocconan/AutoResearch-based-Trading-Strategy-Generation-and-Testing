#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Breakout at weekly Camarilla R1/S1 levels with 1-week trend filter and volume confirmation on 12h timeframe.
Focus on high-probability breakouts during strong trends, using weekly timeframe for trend and weekly pivots for structure.
Designed to work in both bull and bear markets by following the weekly trend and requiring volume confirmation to avoid false breakouts.
Target: 12-37 trades per year (50-150 over 4 years) to minimize fee drag.
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
    
    # Get 1w data for weekly trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    w1_uptrend = ema_50_1w > np.roll(ema_50_1w, 1)  # Rising EMA = uptrend
    w1_downtrend = ema_50_1w < np.roll(ema_50_1w, 1)  # Falling EMA = downtrend
    
    # Calculate weekly Camarilla R1 and S1 from previous week
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    # Camarilla R1 and S1 levels (using previous week's range)
    R1 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 12
    S1 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 12
    
    # Align all weekly data to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    w1_uptrend_aligned = align_htf_to_ltf(prices, df_1w, w1_uptrend)
    w1_downtrend_aligned = align_htf_to_ltf(prices, df_1w, w1_downtrend)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        # Long: price breaks above R1 + weekly uptrend + volume surge
        long_entry = (close[i] > R1_aligned[i] and 
                     w1_uptrend_aligned[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S1 + weekly downtrend + volume surge
        short_entry = (close[i] < S1_aligned[i] and 
                      w1_downtrend_aligned[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < S1_aligned[i] and volume_surge[i]
        short_exit = close[i] > R1_aligned[i] and volume_surge[i]
        
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

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0