#!/usr/bin/env python3
"""
6h_1d_donchian_weekly_pivot_trend - Novel 6h strategy combining Donchian breakouts with weekly pivot trend filter
Hypothesis: On 6h timeframe, price breaks of 20-period Donchian channels are filtered by weekly pivot direction.
In bullish weekly regime (price above weekly pivot), only long Donchian breakouts are taken.
In bearish weekly regime (price below weekly pivot), only short Donchian breakouts are taken.
This avoids counter-trend trades in strong weekly trends and reduces false breakouts.
Target: 15-25 trades/year (60-100 total over 4 years) to stay well under frequency limits.
Works in both bull/bear markets by adapting to weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === DAILY DONCHIAN CHANNEL (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels using previous 20 days (no look-ahead)
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === WEEKLY PIVOT TREND FILTER ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot using previous week's data
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Determine weekly trend: 1 = bullish (price above pivot), -1 = bearish (price below pivot)
    weekly_trend = np.full_like(close, 0, dtype=np.int8)
    for i in range(len(close)):
        if not np.isnan(weekly_pivot_6h[i]):
            if close[i] > weekly_pivot_6h[i]:
                weekly_trend[i] = 1  # bullish weekly regime
            elif close[i] < weekly_pivot_6h[i]:
                weekly_trend[i] = -1  # bearish weekly regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(weekly_pivot_6h[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_6h[i]
        short_breakout = close[i] < donchian_low_6h[i]
        
        # Weekly trend filter: only take trades in direction of weekly trend
        long_signal = long_breakout and weekly_trend[i] == 1
        short_signal = short_breakout and weekly_trend[i] == -1
        
        # Exit on opposite Donchian touch (reduces whipsaw)
        long_exit = close[i] < donchian_low_6h[i]
        short_exit = close[i] > donchian_high_6h[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_weekly_pivot_trend"
timeframe = "6h"
leverage = 1.0