#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (weekly EMA50)
# Long when price breaks above Donchian(20) high and weekly EMA50 is rising
# Short when price breaks below Donchian(20) low and weekly EMA50 is falling
# Uses weekly trend to avoid counter-trend trades. Target: 30-100 total trades over 4 years.

name = "1d_donchian20_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian Channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA50 slope
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_slope = np.diff(ema_1w, prepend=ema_1w[0])  # simple slope approximation
    ema_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly trend data not available
        if np.isnan(ema_1w_slope_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or Donchian opposite touch
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or ema_1w_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or ema_1w_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter
            # Long: price breaks above Donchian high with rising weekly EMA
            if close[i] > donchian_high[i] and ema_1w_slope_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with falling weekly EMA
            elif close[i] < donchian_low[i] and ema_1w_slope_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
    
    return signals