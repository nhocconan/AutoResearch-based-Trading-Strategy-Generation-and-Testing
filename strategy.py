#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation.
- Long when price breaks above Donchian(20) high AND close > 1d EMA34 AND volume > 1.5 * 20-period average
- Short when price breaks below Donchian(20) low AND close < 1d EMA34 AND volume > 1.5 * 20-period average
- Exit when price touches opposite Donchian(10) level (e.g., long exit at Donchian(10) low)
- Uses 4h primary with 1d HTF for EMA trend filter to avoid counter-trend trades
- Donchian channels provide objective breakout levels; EMA34 filters for primary trend; volume confirms conviction
- Designed to capture sustained moves in both bull and bear markets with trend alignment
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year)
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
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels for breakout
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
    
    donchian_high_20 = rolling_max(high, 20)
    donchian_low_20 = rolling_min(low, 20)
    donchian_high_10 = rolling_max(high, 10)
    donchian_low_10 = rolling_min(low, 10)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 34)  # Need Donchian20, volume MA, and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND close > 1d EMA34 AND volume confirmation
            if close[i] > donchian_high_20[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low AND close < 1d EMA34 AND volume confirmation
            elif close[i] < donchian_low_20[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches Donchian(10) low (trailing exit)
            if close[i] <= donchian_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches Donchian(10) high (trailing exit)
            if close[i] >= donchian_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0