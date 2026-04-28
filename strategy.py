#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Momentum_v1
Hypothesis: Trade weekly pivot point breakouts with momentum confirmation on daily timeframe.
Designed to work in both bull and bear markets by requiring breakout momentum and
volume confirmation, avoiding false breakouts in ranging conditions. Targets 10-20 trades/year
to minimize fee drag while capturing sustained directional moves.
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    
    # Weekly pivot point and support/resistance levels
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    r1 = 2 * pivot - prev_weekly_low
    s1 = 2 * pivot - prev_weekly_high
    r2 = pivot + (prev_weekly_high - prev_weekly_low)
    s2 = pivot - (prev_weekly_high - prev_weekly_low)
    
    # Align weekly pivot levels to daily
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Momentum confirmation: price > 20-day EMA for longs, < 20-day EMA for shorts
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_20[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with momentum and volume confirmation
        # Long: price breaks above R1 + price above EMA20 + volume surge
        long_entry = (close[i] > r1_aligned[i] and 
                     close[i] > ema_20[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S1 + price below EMA20 + volume surge
        short_entry = (close[i] < s1_aligned[i] and 
                      close[i] < ema_20[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < s1_aligned[i] and volume_surge[i]
        short_exit = close[i] > r1_aligned[i] and volume_surge[i]
        
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

name = "1d_WeeklyPivot_Breakout_Momentum_v1"
timeframe = "1d"
leverage = 1.0