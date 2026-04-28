#!/usr/bin/env python3
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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly high/low for trend filter
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Daily data for pivot points
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate weekly trend (higher highs and higher lows for uptrend)
    # Using 4-week lookback for trend determination
    weekly_high_ma = pd.Series(weekly_high).rolling(window=4, min_periods=4).mean()
    weekly_low_ma = pd.Series(weekly_low).rolling(window=4, min_periods=4).mean()
    
    weekly_uptrend = weekly_high_ma > weekly_high_ma.shift(1)
    weekly_downtrend = weekly_low_ma < weekly_low_ma.shift(1)
    
    # Align weekly trend to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.values)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.values)
    
    # Calculate Camarilla pivot levels from previous day
    # Classic Camarilla formula
    camarilla_H4 = daily_close + 1.1 * (daily_high - daily_low) * 1.1 / 2
    camarilla_L4 = daily_close - 1.1 * (daily_high - daily_low) * 1.1 / 2
    camarilla_H3 = daily_close + 1.1 * (daily_high - daily_low) * 1.1 / 4
    camarilla_L3 = daily_close - 1.1 * (daily_high - daily_low) * 1.1 / 4
    camarilla_H2 = daily_close + 1.1 * (daily_high - daily_low) * 1.1 / 6
    camarilla_L2 = daily_close - 1.1 * (daily_high - daily_low) * 1.1 / 6
    camarilla_H1 = daily_close + 1.1 * (daily_high - daily_low) * 1.1 / 12
    camarilla_L1 = daily_close - 1.1 * (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Volume confirmation: current volume > 1.8x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla H4/L4 levels
        breakout_up = close[i] > camarilla_H4_aligned[i]
        breakout_down = close[i] < camarilla_L4_aligned[i]
        
        # Pullback conditions at Camarilla H3/L3 levels (mean reversion in strong trend)
        pullback_long = close[i] < camarilla_L3_aligned[i] and weekly_uptrend_aligned[i]
        pullback_short = close[i] > camarilla_H3_aligned[i] and weekly_downtrend_aligned[i]
        
        # Volume filter
        vol_filter = volume_surge[i]
        
        # Entry conditions
        # Long: weekly uptrend + breakout above H4 OR pullback to L3 in uptrend
        long_entry = (breakout_up and weekly_uptrend_aligned[i] and vol_filter) or \
                     (pullback_long and vol_filter)
        # Short: weekly downtrend + breakdown below L4 OR pullback to H3 in downtrend
        short_entry = (breakout_down and weekly_downtrend_aligned[i] and vol_filter) or \
                      (pullback_short and vol_filter)
        
        # Exit conditions: opposite signal or loss of weekly trend
        long_exit = not weekly_uptrend_aligned[i] or (close[i] < camarilla_L1_aligned[i] and weekly_uptrend_aligned[i])
        short_exit = not weekly_downtrend_aligned[i] or (close[i] > camarilla_H1_aligned[i] and weekly_downtrend_aligned[i])
        
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

name = "6h_Camarilla_H4L4_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0