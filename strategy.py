#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    1d Donchian channel breakout with weekly trend filter.
    - Long: Close breaks above weekly Donchian high (20-period) and weekly MA > prior MA
    - Short: Close breaks below weekly Donchian low (20-period) and weekly MA < prior MA
    - Exit: Opposite breakout
    - Uses weekly Donchian channels (20-period) for breakout levels
    - Weekly MA filter prevents counter-trend trades
    - Target: 10-25 trades/year on 1d timeframe
    """
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Donchian channels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly high and low arrays
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate 20-period Donchian channels on weekly data
    # Donchian high = max of last 20 weekly highs
    # Donchian low = min of last 20 weekly lows
    from collections import deque
    high_max = deque(maxlen=20)
    low_min = deque(maxlen=20)
    
    # Initialize deques
    for i in range(min(20, len(weekly_high))):
        high_max.append(weekly_high[i])
        low_min.append(weekly_low[i])
    
    weekly_dc_high = np.full(len(weekly_high), np.nan)
    weekly_dc_low = np.full(len(weekly_low), np.nan)
    
    for i in range(len(weekly_high)):
        if i >= 20:
            high_max.append(weekly_high[i])
            low_min.append(weekly_low[i])
            weekly_dc_high[i] = max(high_max) if len(high_max) == 20 else np.nan
            weekly_dc_low[i] = min(low_min) if len(low_min) == 20 else np.nan
        elif i < 20:
            # Still building history
            weekly_dc_high[i] = max(weekly_high[:i+1]) if i >= 0 else np.nan
            weekly_dc_low[i] = min(weekly_low[:i+1]) if i >= 0 else np.nan
    
    # Calculate weekly MA for trend filter (20-period SMA)
    weekly_sma = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    weekly_dc_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_dc_high)
    weekly_dc_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_dc_low)
    weekly_sma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if np.isnan(weekly_dc_high_aligned[i]) or np.isnan(weekly_dc_low_aligned[i]) or np.isnan(weekly_sma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly Donchian high AND weekly MA rising
            if close[i] > weekly_dc_high_aligned[i] and weekly_sma_aligned[i] > weekly_sma_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low AND weekly MA falling
            elif close[i] < weekly_dc_low_aligned[i] and weekly_sma_aligned[i] < weekly_sma_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Break below weekly Donchian low
            if close[i] < weekly_dc_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Break above weekly Donchian high
            if close[i] > weekly_dc_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals