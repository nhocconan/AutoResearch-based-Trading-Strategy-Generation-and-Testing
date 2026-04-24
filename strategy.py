#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Long when close > Donchian upper(20) AND 1w EMA50 rising AND volume > 1.5 * 20-period average
- Short when close < Donchian lower(20) AND 1w EMA50 falling AND volume > 1.5 * 20-period average
- Exit when price crosses Donchian midpoint OR volume drops below average
- Uses 12h primary with 1w HTF for trend filter to avoid counter-trend trades
- Donchian captures breakouts, EMA50 filters trend direction, volume confirms conviction
- Designed to work in both bull (upward breakouts) and bear (downward breakouts) markets
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate Donchian channels (20-period)
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
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    ema_rising = ema_50_slope > 0
    ema_falling = ema_50_slope < 0
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1  # Need Donchian and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > Donchian upper AND EMA50 rising AND volume confirmation
            if close[i] > donchian_upper[i] and ema_rising[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < Donchian lower AND EMA50 falling AND volume confirmation
            elif close[i] < donchian_lower[i] and ema_falling[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close < Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close > Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0