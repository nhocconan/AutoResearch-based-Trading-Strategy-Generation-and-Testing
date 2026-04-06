#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly Trend Filter
Hypothesis: Daily Donchian breakouts capture strong directional moves. Weekly trend filter (EMA50) ensures alignment with higher timeframe trend, while volume confirmation validates breakout strength. This reduces false signals and improves performance in both bull and bear markets by avoiding counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = close_weekly[i] * (2 / 51) + ema50_weekly[i-1] * (49 / 51)
    
    # Align weekly EMA50 to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period moving average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50)  # For Donchian and weekly EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema50_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price crosses below/above weekly EMA50
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA50
            if close[i] < donchian_low[i] or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA50
            if close[i] > donchian_high[i] or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5  # Volume 1.5x average
            # Long only if price above weekly EMA50 (uptrend), short only if below (downtrend)
            trend_align_long = close[i] > ema50_weekly_aligned[i]
            trend_align_short = close[i] < ema50_weekly_aligned[i]
            
            if bull_breakout and volume_filter and trend_align_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and trend_align_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals