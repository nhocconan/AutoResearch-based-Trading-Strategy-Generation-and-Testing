#!/usr/bin/env python3
"""
12h_WeeklyDonchianBreakout_TrendFilter
Hypothesis: Trade 12h timeframe using weekly Donchian breakouts (20-period) filtered by daily trend (price above/below EMA50). 
Long when price breaks above weekly Donchian high and daily close > daily EMA50; short when price breaks below weekly Donchian low and daily close < daily EMA50. 
Exit when price crosses back through the weekly Donchian midpoint. 
Uses weekly structure for trend and daily filter to avoid counter-trend trades. Designed for low frequency (target 15-30 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "12h_WeeklyDonchianBreakout_TrendFilter"
timeframe = "12h"
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
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    donchian_high = np.full_like(high_weekly, np.nan)
    donchian_low = np.full_like(low_weekly, np.nan)
    
    for i in range(20, len(high_weekly)):
        donchian_high[i] = np.max(high_weekly[i-20:i])
        donchian_low[i] = np.min(low_weekly[i-20:i])
    
    # Align Donchian levels to 12h timeframe (with 1-bar delay for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid = (donchian_high_aligned + donchian_low_aligned) / 2.0
    
    # Get daily data for trend filter (EMA50)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    close_daily_aligned = align_htf_to_ltf(prices, df_daily, close_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for daily EMA50 and weekly Donchian
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_daily_aligned[i]) or 
            np.isnan(close_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND daily close > daily EMA50
            if close[i] > donchian_high_aligned[i] and close_daily_aligned[i] > ema_50_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low AND daily close < daily EMA50
            elif close[i] < donchian_low_aligned[i] and close_daily_aligned[i] < ema_50_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals