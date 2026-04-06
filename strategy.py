#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly Trend Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume confirmation ensures
breakout strength, while weekly trend filter avoids counter-trend trades. Designed for
30-100 trades over 4 years (7-25/year) to minimize fee drag while adapting to bull/bear
markets via weekly trend filter.
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
    
    # Weekly EMA 50 for trend direction
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 50:
        ema_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 / 51) + (ema_weekly[i-1] * 49 / 51)
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
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
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or weekly trend reversal
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly trend alignment
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            bullish_trend = close[i] > ema_weekly_aligned[i]
            bearish_trend = close[i] < ema_weekly_aligned[i]
            
            if bull_breakout and volume_filter and bullish_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter and bearish_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals