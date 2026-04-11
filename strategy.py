#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with Weekly Trend Filter
Hypothesis: Breakouts in the direction of the weekly trend capture strong momentum moves
while avoiding counter-trend whipsaws. Weekly trend filter reduces false breakouts in ranging
markets. Works in both bull (breakouts continue up) and bear (breakouts continue down) markets.
Target: 50-150 trades over 4 years with strict entry conditions.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high: np.ndarray, low: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Donchian channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema_50_1w  # True when in uptrend
    
    # Align weekly trend to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    # Daily data for volume context (optional filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 6h data
    upper, lower = calculate_donchian(high, low, window=20)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_uptrend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i]   # Break above upper band
        breakdown_down = close[i] < lower[i]  # Break below lower band
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_trend_up = weekly_uptrend_aligned[i] > 0.5
        weekly_trend_down = weekly_uptrend_aligned[i] <= 0.5
        
        # Entry conditions: breakout in direction of weekly trend + volume
        long_entry = breakout_up and weekly_trend_up and volume_filter
        short_entry = breakdown_down and weekly_trend_down and volume_filter
        
        # Exit conditions: return to middle of Donchian channel
        mid_channel = (upper[i] + lower[i]) / 2
        long_exit = close[i] < mid_channel
        short_exit = close[i] > mid_channel
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals