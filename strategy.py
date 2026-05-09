#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w trend filter and 12h Donchian breakout.
# Uses 1w EMA50 for trend filter and 12h Donchian(20) breakout for entries.
# Designed for low trade frequency (12-37/year) to avoid fee drag in 12h timeframe.
# Works in both bull/bear markets by requiring alignment with 1w trend and breakout confirmation.
name = "12h_Donchian20_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):  # Need 20 bars for Donchian
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 19)  # Wait for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_low[i-1]  # Break below previous low
        
        if position == 0:
            # Long: price above 1w EMA50 and breakout above Donchian high
            if close[i] > ema_50_12h[i] and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA50 and breakout below Donchian low
            elif close[i] < ema_50_12h[i] and breakout_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1w EMA50 or breakdown below Donchian low
            if close[i] < ema_50_12h[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1w EMA50 or breakout above Donchian high
            if close[i] > ema_50_12h[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals