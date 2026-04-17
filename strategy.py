#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above Donchian(20) upper band AND volume > 1.5x 20-period average AND 1d ADX > 25.
Short when price breaks below Donchian(20) lower band AND volume > 1.5x 20-period average AND 1d ADX > 25.
Exit when price reverses to Donchian midpoint OR volume drops below average.
Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
Works in both bull and bear markets by requiring strong trend (ADX > 25) for entries.
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
    volume = prices['volume'].values
    
    # Get 12h data for Donchian and volume (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for ADX (HTF trend filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian Channel (20-period) on 12h
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_window = 20
    upper_12h = rolling_max(high_12h, donchian_window)
    lower_12h = rolling_min(low_12h, donchian_window)
    middle_12h = (upper_12h + lower_12h) / 2.0
    
    # Calculate ADX (14-period) on 1d
    def wilders_smoothing(arr, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period - 1] = np.mean(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM
    tr_period = 14
    atr_1d = wilders_smoothing(tr, tr_period)
    plus_dm_1d = wilders_smoothing(plus_dm, tr_period)
    minus_dm_1d = wilders_smoothing(minus_dm, tr_period)
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_1d / atr_1d
    minus_di_1d = 100 * minus_dm_1d / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, tr_period)
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx = adx_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above upper band AND volume > 1.5x avg AND ADX > 25 (strong trend)
            if price > upper and vol > 1.5 * vol_ma and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band AND volume > 1.5x avg AND ADX > 25 (strong trend)
            elif price < lower and vol > 1.5 * vol_ma and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reverses to middle band OR volume drops below average
            if price < middle or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reverses to middle band OR volume drops below average
            if price > middle or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0