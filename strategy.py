#!/usr/bin/env python3
# 6h_1w_donchian_20_breakout_volume_v1
# Strategy: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture institutional breakout moves. Weekly trend filter ensures alignment with major trend, reducing false breakouts. Volume confirms breakout sincerity. Designed for low trade frequency to minimize fee drag in 2025 bear market.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_20_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly close for trend filter (no indicator needed, just price trend)
    close_1w = df_1w['close'].values
    # Weekly trend: price above/below prior weekly close (simple trend)
    weekly_trend_up = np.zeros(len(close_1w), dtype=bool)
    weekly_trend_down = np.zeros(len(close_1w), dtype=bool)
    for i in range(1, len(close_1w)):
        weekly_trend_up[i] = close_1w[i] > close_1w[i-1]
        weekly_trend_down[i] = close_1w[i] < close_1w[i-1]
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # 6h Donchian Channel (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: Donchian breakout + volume + weekly trend alignment
        if close[i] > highest_high[i] and vol_confirm[i] and weekly_trend_up_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < lowest_low[i] and vol_confirm[i] and weekly_trend_down_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian breakout (reversal signal)
        elif position == 1 and close[i] < lowest_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > highest_high[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals