#!/usr/bin/env python3
"""
1d_WeeklyPivot_R4_S4_Breakout_1wTrend_Volume
Hypothesis: Daily chart breakouts at weekly-derived Camarilla R4/S4 levels with weekly trend filter and volume confirmation.
Targets 7-25 trades/year by requiring breakout, weekly trend alignment, and volume surge to reduce false signals.
Works in both bull and bear markets by using trend filter and volatility-based entries.
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
    
    # Get weekly data for Camarilla levels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    prev_high = df_weekly['high'].shift(1).values
    prev_low = df_weekly['low'].shift(1).values
    prev_close = df_weekly['close'].shift(1).values
    
    # Camarilla R4 and S4 levels (widest bands)
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Weekly EMA20 for trend filter
    ema_20_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    R4_aligned = align_htf_to_ltf(prices, df_weekly, R4)
    S4_aligned = align_htf_to_ltf(prices, df_weekly, S4)
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume confirmation: current volume > 2.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_20_weekly_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        # Long: price breaks above R4 + weekly uptrend + volume surge
        long_entry = (close[i] > R4_aligned[i] and 
                     close[i] > ema_20_weekly_aligned[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S4 + weekly downtrend + volume surge
        short_entry = (close[i] < S4_aligned[i] and 
                      close[i] < ema_20_weekly_aligned[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < S4_aligned[i] and volume_surge[i]
        short_exit = close[i] > R4_aligned[i] and volume_surge[i]
        
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

name = "1d_WeeklyPivot_R4_S4_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0