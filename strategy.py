#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Filter
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter capture strong momentum shifts while avoiding false breakouts in ranging markets. Weekly trend ensures alignment with higher timeframe direction, reducing whipsaws. Designed for low trade frequency (~10-20/year) to work in both bull and bear markets via trend alignment and discrete position sizing (0.25) to minimize fee drag.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate EMA20 on 1w close for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels on daily data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window_high = high[i - lookback + 1:i + 1]
        window_low = low[i - lookback + 1:i + 1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)
    
    # Align HTF EMA20 to 1d timeframe (standard 1-bar delay for EMA)
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20)
    start_idx = lookback - 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Donchian breakout signals with weekly trend filter
            # Long: price breaks above Donchian upper in uptrend (close > EMA20_1w)
            # Short: price breaks below Donchian lower in downtrend (close < EMA20_1w)
            long_signal = close[i] > highest_high[i] and close[i] > ema20_aligned[i]
            short_signal = close[i] < lowest_low[i] and close[i] < ema20_aligned[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian lower (mean reversion)
            exit_signal = close[i] < lowest_low[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian upper (mean reversion)
            exit_signal = close[i] > highest_high[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0