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
    
    # Get weekly data for Donchian breakout and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly upper/lower band (20-period high/low)
    weekly_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (using previous weekly bar's values)
    weekly_high_daily = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_daily = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Weekly trend filter: 50-period EMA on weekly close
    close_weekly = df_weekly['close'].values
    weekly_ema50 = pd.Series(close_weekly).ewm(span=50, min_periods=50).mean().values
    weekly_ema50_daily = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Volume filter: current volume > 2.0x 20-period average (tightened for 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_daily[i]) or np.isnan(weekly_low_daily[i]) or 
            np.isnan(weekly_ema50_daily[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high AND above weekly EMA50 (uptrend) AND volume surge
            if close[i] > weekly_high_daily[i] and close[i] > weekly_ema50_daily[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low AND below weekly EMA50 (downtrend) AND volume surge
            elif close[i] < weekly_low_daily[i] and close[i] < weekly_ema50_daily[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly low OR below weekly EMA50 (trend change)
            if close[i] < weekly_low_daily[i] or close[i] < weekly_ema50_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly high OR above weekly EMA50 (trend change)
            if close[i] > weekly_high_daily[i] or close[i] > weekly_ema50_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals