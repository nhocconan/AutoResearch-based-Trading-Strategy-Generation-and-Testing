#!/usr/bin/env python3
"""
6h_Aroon_Consolidation_Breakout_12hTrend
Hypothesis: Aroon indicator identifies consolidation periods (Aroon Up & Down < 30).
Breakouts from Aroon-defined consolidation with 12-hour EMA50 trend filter capture
explosive moves while avoiding false breakouts. Works in bull/bear by filtering
with higher timeframe trend. Targets 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Aroon indicator (25-period) to identify consolidation
    # Aroon Up = ((25 - periods since highest high) / 25) * 100
    # Aroon Down = ((25 - periods since lowest low) / 25) * 100
    aroon_period = 25
    
    # Calculate periods since highest high and lowest low
    def periods_since_extreme(arr, is_high):
        n = len(arr)
        since_extreme = np.full(n, np.nan)
        for i in range(arood_period, n):
            window = arr[i-arood_period:i+1]
            if is_high:
                ext_idx = np.argmax(window)
            else:
                ext_idx = np.argmin(window)
            since_extreme[i] = aroon_period - ext_idx
        return since_extreme
    
    # Vectorized approach using pandas rolling
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Time since highest high in last aroon_period periods
    since_high = high_series.rolling(aroon_period, min_periods=1).apply(
        lambda x: aroon_period - np.argmax(x) if len(x) > 0 else np.nan, raw=True
    ).values
    
    # Time since lowest low in last aroon_period periods
    since_low = low_series.rolling(aroon_period, min_periods=1).apply(
        lambda x: aroon_period - np.argmin(x) if len(x) > 0 else np.nan, raw=True
    ).values
    
    aroon_up = ((aroon_period - since_high) / aroon_period) * 100
    aroon_down = ((aroon_period - since_low) / aroon_period) * 100
    
    # Consolidation: both Aroon Up and Down below 30 (weak trend in both directions)
    consolidation = (aroon_up < 30) & (aroon_down < 30)
    
    # Breakout bands: highest high and lowest low during consolidation period
    # We'll use the highest high and lowest low from the last 25 periods as breakout levels
    highest_high = pd.Series(high).rolling(window=aroon_period, min_periods=aroon_period).max().values
    lowest_low = pd.Series(low).rolling(window=aroon_period, min_periods=aroon_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = aroon_period * 2  # Need enough data for Aroon calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(aroon_up[i]) or
            np.isnan(aroon_down[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Consolidation check (must be consolidating before breakout)
        is_consolidating = consolidation[i-1]  # Use previous bar to avoid lookahead
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above recent high
        breakout_down = close[i] < lowest_low[i-1]  # Break below recent low
        
        # Entry: breakout from consolidation in direction of trend
        long_entry = is_consolidating and uptrend and breakout_up
        short_entry = is_consolidating and downtrend and breakout_down
        
        # Exit: opposite breakout or trend change
        long_exit = breakout_down or (not uptrend)
        short_exit = breakout_up or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Aroon_Consolidation_Breakout_12hTrend"
timeframe = "6h"
leverage = 1.0