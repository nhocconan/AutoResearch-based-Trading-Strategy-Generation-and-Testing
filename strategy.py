#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Long when price breaks above Donchian upper channel (20-period high) AND 1d ATR(14) > 20-period SMA of ATR AND volume > 1.5 * 20-period average
- Short when price breaks below Donchian lower channel (20-period low) AND same filters
- Exit when price returns to Donchian midpoint (average of upper/lower) OR ATR filter fails
- Uses 12h primary with 1d HTF for ATR regime filter to avoid low-volatility false breakouts
- Donchian captures structural breaks; ATR filter ensures sufficient volatility; volume confirms conviction
- Designed to work in both bull (upward breaks) and bear (downward breaks) markets with volatility filter
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
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    midpoint = (upper_channel + lower_channel) / 2
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volatility filter: ATR > ATR moving average (expanding volatility)
    vol_filter = atr_1d_aligned > atr_ma_1d_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20, 30) + 5  # Need Donchian, volume MA, and ATR data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel AND volatility filter AND volume confirmation
            if close[i] > upper_channel[i] and vol_filter[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND volatility filter AND volume confirmation
            elif close[i] < lower_channel[i] and vol_filter[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to midpoint OR volatility filter fails
            if close[i] <= midpoint[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint OR volatility filter fails
            if close[i] >= midpoint[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0