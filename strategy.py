#!/usr/bin/env python3
"""
1D_Weekly_Camarilla_R1_S1_Breakout_TrendFilter
Hypothesis: Use weekly trend (price above/below 20-week EMA) to filter daily breakouts at Camarilla R1/S1 levels.
Long when price crosses above daily EMA(50) and touches daily R1 in uptrend; 
Short when price crosses below daily EMA(50) and touches daily S1 in downtrend.
Volume confirmation: current volume > 1.5x 20-day average volume.
Designed for low frequency (target 10-30 trades/year) to work in both bull and bear markets by aligning with weekly trend.
"""
name = "1D_Weekly_Camarilla_R1_S1_Breakout_TrendFilter"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend
    close_weekly = pd.Series(df_weekly['close'])
    ema_weekly = close_weekly.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily EMA(50) for entry trigger
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Camarilla levels (R1, S1)
    # Use prior day's OHLC for today's levels (no look-ahead)
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close[0] = close[0]  # first bar uses current
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    
    pivot = (prior_high + prior_low + prior_close) / 3
    range_prior = prior_high - prior_low
    r1 = pivot + (range_prior * 1.1 / 12)
    s1 = pivot - (range_prior * 1.1 / 12)
    
    # Volume filter: current volume > 1.5 * 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = 50  # Ensure sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 5 days between trades to reduce frequency
            if bars_since_exit < 5:
                continue
                
            # Determine weekly trend: price above/below weekly EMA20
            weekly_uptrend = close[i] > ema_weekly_aligned[i]
            weekly_downtrend = close[i] < ema_weekly_aligned[i]
            
            # Long: price crosses above EMA50 and touches R1 in uptrend
            if (weekly_uptrend and 
                close[i] > ema_50[i] and close[i-1] <= ema_50[i-1] and 
                low[i] <= r1[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below EMA50 and touches S1 in downtrend
            elif (weekly_downtrend and 
                  close[i] < ema_50[i] and close[i-1] >= ema_50[i-1] and 
                  high[i] >= s1[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA50 side
            if position == 1 and close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals