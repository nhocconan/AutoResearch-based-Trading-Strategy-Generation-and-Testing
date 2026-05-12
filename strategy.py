#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data for Donchian breakout and trend filter
    df_week = get_htf_data(prices, '1w')
    
    # Weekly Donchian breakout levels (20-week high/low)
    high_week = df_week['high'].values
    low_week = df_week['low'].values
    
    # Calculate 20-week high and low
    high_20w = pd.Series(high_week).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_week).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(10) for trend filter
    close_week = df_week['close'].values
    ema10_week = pd.Series(close_week).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align weekly indicators to daily timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_week, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_week, low_20w)
    ema10_week_aligned = align_htf_to_ltf(prices, df_week, ema10_week)
    
    # Daily volume spike: current volume > 2.0x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or 
            np.isnan(ema10_week_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-week high + weekly uptrend + volume spike
            if (close[i] > high_20w_aligned[i] and 
                close[i] > ema10_week_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-week low + weekly downtrend + volume spike
            elif (close[i] < low_20w_aligned[i] and 
                  close[i] < ema10_week_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below 20-week low
            if close[i] < low_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above 20-week high
            if close[i] > high_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals