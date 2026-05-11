#!/usr/bin/env python3
name = "1d_WeeklyBreakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_series = pd.Series(close_1w).ewm(span=50, min_periods=50).mean()
    ema_1w = ema_1w_series.values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily Donchian(20) breakout levels
    # Use previous day's high/low to avoid look-ahead
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    
    # Calculate rolling max/min of previous 20 days
    from collections import deque
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        dq = deque()
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                dq.clear()
                continue
            while dq and arr[dq[-1]] <= arr[i]:
                dq.pop()
            dq.append(i)
            if dq[0] <= i - window:
                dq.popleft()
            if i >= window - 1:
                result[i] = arr[dq[0]]
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        dq = deque()
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                dq.clear()
                continue
            while dq and arr[dq[-1]] >= arr[i]:
                dq.pop()
            dq.append(i)
            if dq[0] <= i - window:
                dq.popleft()
            if i >= window - 1:
                result[i] = arr[dq[0]]
        return result
    
    donchian_high = rolling_max(high_shift, 20)
    donchian_low = rolling_min(low_shift, 20)
    
    # Volume filter: current volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly EMA50 (uptrend) AND volume spike
            if close[i] > donchian_high[i] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly EMA50 (downtrend) AND volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Donchian low OR below weekly EMA50 (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above Donchian high OR above weekly EMA50 (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals