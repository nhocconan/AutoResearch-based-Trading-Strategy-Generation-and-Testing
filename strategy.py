#!/usr/bin/env python3
name = "6h_WeeklyDonchianBreakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for Donchian channels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_max = pd.Series(df_weekly['high'].values).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(df_weekly['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align weekly channels to 6h timeframe (wait for weekly close)
    high_max_aligned = align_htf_to_ltf(prices, df_weekly, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_weekly, low_min)
    
    # Weekly trend: EMA50 on weekly close
    ema_50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max_aligned[i]) or 
            np.isnan(low_min_aligned[i]) or 
            np.isnan(ema_50_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        trend_up = close > ema_50_weekly_aligned[i]
        trend_down = close < ema_50_weekly_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above weekly Donchian high in uptrend
            if (close[i] > high_max_aligned[i] and 
                trend_up[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below weekly Donchian low in downtrend
            elif (close[i] < low_min_aligned[i] and 
                  trend_down[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters weekly Donchian range or trend change
            if (close[i] < high_max_aligned[i] and close[i] > low_min_aligned[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters weekly Donchian range or trend change
            if (close[i] < high_max_aligned[i] and close[i] > low_min_aligned[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using 6h timeframe with weekly Donchian breakout (20-period) and weekly EMA50 trend filter
# will capture major trend continuations in both bull and bear markets. The weekly timeframe provides
# a robust trend filter that avoids whipsaws, while the 6h timeframe allows timely entry on breakouts.
# Position size of 0.25 manages drawdown, and cooldown of 4 bars prevents overtrading. This strategy
# targets 20-50 total trades over 4 years (5-12/year) to minimize fee drag. The weekly Donchian
# channels act as strong support/resistance levels that institutions watch, making breakouts significant.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).