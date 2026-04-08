#!/usr/bin/env python3
# 6h_donchian_weekly_pivot_volume_v1
# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly pivot > prior week pivot (uptrust) AND volume > 1.5x average.
# Short when price breaks below Donchian(20) low AND weekly pivot < prior week pivot (downtrend) AND volume > 1.5x average.
# Uses weekly pivot to capture multi-week trend, Donchian for breakout, volume to avoid fakeouts.
# Targets 15-30 trades/year by requiring confluence of three filters. Works in bull/bear by following weekly trend.

name = "6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Breakout signals
    breakout_up = close > donchian_high  # price above prior 20-period high
    breakout_down = close < donchian_low  # price below prior 20-period low
    
    # Volume filter: 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_ma[:19] = vol_ma[19] if not np.isnan(vol_ma[19]) else 0
    
    volume_filter = volume > 1.5 * vol_ma
    
    # Get weekly data for trend filter (pivot trend)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    
    # Weekly pivot trend: current pivot > prior week pivot = uptrend
    pivot_trend_up = np.zeros(len(pivot_weekly), dtype=bool)
    pivot_trend_down = np.zeros(len(pivot_weekly), dtype=bool)
    for i in range(1, len(pivot_weekly)):
        pivot_trend_up[i] = pivot_weekly[i] > pivot_weekly[i-1]
        pivot_trend_down[i] = pivot_weekly[i] < pivot_weekly[i-1]
    
    # Align weekly pivot trend to 6h timeframe
    pivot_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, pivot_trend_up)
    pivot_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, pivot_trend_down)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(lookback, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0 or
            np.isnan(pivot_trend_up_aligned[i]) or np.isnan(pivot_trend_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit if breakout fails or trend reverses
            if not breakout_up[i] or not pivot_trend_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if breakout fails or trend reverses
            if not breakout_down[i] or not pivot_trend_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish breakout, volume confirmation, and weekly uptrend
            if (breakout_up[i] and 
                volume_filter[i] and 
                pivot_trend_up_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish breakout, volume confirmation, and weekly downtrend
            elif (breakout_down[i] and 
                  volume_filter[i] and 
                  pivot_trend_down_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals