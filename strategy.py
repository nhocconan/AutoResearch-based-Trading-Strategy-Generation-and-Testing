#!/usr/bin/env python3
# 1D_WeeklyDonchian_Breakout_WickFilter_Volume
# Hypothesis: On 1d timeframe, enter long when price breaks above weekly Donchian high (20) with bullish weekly close and volume confirmation.
# Short when price breaks below weekly Donchian low (20) with bearish weekly close and volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades and Donchian levels from weekly for precise entries.
# Wick filter: require close in upper/lower third of daily range to avoid false breakouts.
# Target: 10-30 trades/year per symbol (40-120 total over 4 years).

name = "1D_WeeklyDonchian_Breakout_WickFilter_Volume"
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
    
    # Get weekly data for Donchian levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: close above/below 50-period EMA
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1w > ema_50
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Wick filter: close in upper third for long, lower third for short
    daily_range = high - low
    upper_third = low + (daily_range * 2/3)
    lower_third = low + (daily_range * 1/3)
    close_upper = close > upper_third
    close_lower = close < lower_third
    
    # Align weekly indicators to daily
    high_max_aligned = align_htf_to_ltf(prices, df_1w, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1w, low_min)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + weekly uptrend + volume confirmation + close in upper third
            if close[i] > high_max_aligned[i] and trend_up_aligned[i] and volume_confirm[i] and close_upper[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + weekly downtrend + volume confirmation + close in lower third
            elif close[i] < low_min_aligned[i] and not trend_up_aligned[i] and volume_confirm[i] and close_lower[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low or trend changes
            if close[i] < low_min_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high or trend changes
            if close[i] > high_max_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals