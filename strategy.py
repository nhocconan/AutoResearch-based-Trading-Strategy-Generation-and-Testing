#!/usr/bin/env python3
name = "1d_WeeklyBreakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter and breakout levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for breakout levels (from previous week)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly range for breakout levels
    range_w = high_w - low_w
    
    # Breakout levels: buy above weekly high, sell below weekly low
    breakout_high = high_w  # weekly high
    breakout_low = low_w    # weekly low
    
    # Align weekly breakout levels to daily timeframe
    breakout_high_d = align_htf_to_ltf(prices, df_w, breakout_high)
    breakout_low_d = align_htf_to_ltf(prices, df_w, breakout_low)
    
    # Weekly EMA40 for trend filter
    close_w_series = pd.Series(close_w)
    ema_w = close_w_series.ewm(span=40, min_periods=40).mean().values
    ema_w_d = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(breakout_high_d[i]) or np.isnan(breakout_low_d[i]) or 
            np.isnan(ema_w_d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high AND above weekly EMA40 (uptrend) AND volume surge
            if close[i] > breakout_high_d[i] and close[i] > ema_w_d[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low AND below weekly EMA40 (downtrend) AND volume surge
            elif close[i] < breakout_low_d[i] and close[i] < ema_w_d[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly low OR below weekly EMA40 (trend change)
            if close[i] < breakout_low_d[i] or close[i] < ema_w_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly high OR above weekly EMA40 (trend change)
            if close[i] > breakout_high_d[i] or close[i] > ema_w_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals