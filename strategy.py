#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with volume confirmation and weekly trend filter.
# Long when price breaks above Donchian(20) high + volume > 1.5x avg volume + weekly uptrend.
# Short when price breaks below Donchian(20) low + volume > 1.5x avg volume + weekly downtrend.
# Exit when price crosses the opposite Donchian band or trend reverses.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull (breakout continuation) and bear (breakdown continuation).

name = "1d_Donchian_20_Volume_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Donchian channels (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly EMA(10) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_10_1w = close_1w_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    trend_1w_up = ema_10_1w[1:] > ema_10_1w[:-1]
    trend_1w_up = np.concatenate([[False], trend_1w_up])
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(trend_1w_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_ratio = volume[i] / avg_volume[i] if avg_volume[i] > 0 else 0
        
        if position == 0:
            # Long breakout: price > Donchian high + volume surge + weekly uptrend
            if close[i] > highest_high[i] and volume_ratio > 1.5 and trend_1w_up_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian low + volume surge + weekly downtrend
            elif close[i] < lowest_low[i] and volume_ratio > 1.5 and not trend_1w_up_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian low or weekly trend turns down
            if close[i] < lowest_low[i] or not trend_1w_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian high or weekly trend turns up
            if close[i] > highest_high[i] or trend_1w_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals