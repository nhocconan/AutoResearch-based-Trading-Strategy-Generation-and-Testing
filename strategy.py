#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Weekly Trend Filter
# Hypothesis: Donchian(20) breakouts in direction of weekly EMA(50) trend capture momentum with minimal trades.
# Weekly trend filter prevents counter-trend trades during reversals. Breakouts provide clear entry/exit.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "12h_donchian_breakout_weekly_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on weekly close
    close_weekly = df_weekly['close'].values
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False).mean().values
    
    # Align weekly EMA to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Donchian(20) channels on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band or trend changes
            if close[i] <= lowest_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band or trend changes
            if close[i] >= highest_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Donchian breakout in direction of weekly trend
            if close[i] > ema_50_aligned[i]:  # Uptrend
                if high[i] > highest_high[i]:  # Break above upper band
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                if low[i] < lowest_low[i]:  # Break below lower band
                    position = -1
                    signals[i] = -0.25
    
    return signals