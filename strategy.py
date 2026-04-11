#!/usr/bin/env python3
# 1d_1w_turtle_soup_v1
# Strategy: 1d Turtle Soup reversal pattern with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Turtle Soup fades false breakouts of 20-day highs/lows. In bull markets, buy false breakdowns below 20-day low; in bear markets, sell false breakouts above 20-day high. Weekly trend filter ensures trades align with higher timeframe direction. Volume confirmation avoids low-liquidity false signals. Low frequency (~10-20/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_turtle_soup_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily 20-period high/low for Turtle Soup
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Turtle Soup entry logic: fade false breakouts
        # Long setup: price breaks below 20-day low but closes back above it (false breakdown)
        if (low[i] < lowest_20[i] and close[i] > lowest_20[i] and 
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        # Short setup: price breaks above 20-day high but closes back below it (false breakout)
        elif (high[i] > highest_20[i] and close[i] < highest_20[i] and 
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price reaches opposite 20-day level or trend change
        elif position == 1 and (high[i] >= highest_20[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= lowest_20[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals