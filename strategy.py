#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ADX regime and Donchian levels.
- Donchian levels from prior 1d: upper = 20-period high, lower = 20-period low.
- Long when price breaks above upper Donchian with volume spike in bull regime (ADX > 25).
- Short when price breaks below lower Donchian with volume spike in bear regime (ADX > 25).
- Volume confirmation: current volume > 1.5x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend regime, in bear via selling breakouts in downtrend regime.
- Uses 1d timeframe for HTF to align with proven winning patterns (Camarilla/Donchian + volume + regime).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for regime filter
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Donchian levels from prior 1d (20-period high/low)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high_1d, 20)
    donchian_lower = rolling_min(low_1d, 20)
    
    # Align 1d indicators to 4h (each 1d bar = 6x 4h bars)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # ADX needs ~34 bars (14+14+14+20 for Donchian), volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade when ADX > 25 (trending regime)
            if adx_aligned[i] > 25:
                # Long when price breaks above upper Donchian with volume spike
                if close[i] > donchian_upper_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks below lower Donchian with volume spike
                elif close[i] < donchian_lower_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX25_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0