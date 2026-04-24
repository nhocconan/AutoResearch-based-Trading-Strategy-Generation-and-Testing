#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss.
- Long when close > Donchian Upper(20) AND close > 1w EMA50
- Short when close < Donchian Lower(20) AND close < 1w EMA50
- Exit when price crosses below Donchian Middle(20) for long OR above for short
- Uses 1d primary with 1w HTF for trend filter to avoid whipsaws in ranging markets
- Donchian channels provide clear structure; EMA50 filters for primary trend
- Designed to work in both bull (breakouts above EMA50) and bear (breakdowns below EMA50) markets
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian Channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_20 = rolling_max(high, 20)
    lower_20 = rolling_min(low, 20)
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need Donchian and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(middle_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper band AND above 1w EMA50 (bullish trend)
            if close[i] > upper_20[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band AND below 1w EMA50 (bearish trend)
            elif close[i] < lower_20[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below middle band (mean reversion)
            if close[i] < middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above middle band (mean reversion)
            if close[i] > middle_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0