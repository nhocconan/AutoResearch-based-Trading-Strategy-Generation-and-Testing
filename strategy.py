#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1w ADX regime filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1w ADX(14) for regime filter (ADX > 25 = trending market, ADX < 20 = ranging market).
- Donchian channels: 20-period high/low breakouts for trend continuation.
- Entry: Long when price breaks above Donchian(20) high AND 1w ADX > 25 AND volume > 1.5 * volume MA(20).
         Short when price breaks below Donchian(20) low AND 1w ADX > 25 AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below Donchian(20) low,
        exit short when price crosses above Donchian(20) high.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to capture strong trends while avoiding choppy markets via ADX filter.
Proven pattern from DB: Donchian breakouts with volume and trend filters show SOL test Sharpe up to 0.63.
Adding 1w ADX regime filter should improve performance on BTC/ETH by avoiding false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for regime filter
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(df_1w_high - df_1w_low)
    tr2 = np.abs(df_1w_high - np.roll(df_1w_close, 1))
    tr3 = np.abs(df_1w_low - np.roll(df_1w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((df_1w_high - np.roll(df_1w_high, 1)) > (np.roll(df_1w_low, 1) - df_1w_low),
                       np.maximum(df_1w_high - np.roll(df_1w_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1w_low, 1) - df_1w_low) > (df_1w_high - np.roll(df_1w_high, 1)),
                        np.maximum(np.roll(df_1w_low, 1) - df_1w_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Rolling max/min for Donchian channels
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Align HTF indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14+13+13, 20)  # Need enough bars for ADX, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and ADX > 25 (trending)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            strong_trend = adx_aligned[i] > 25
            
            # Long: Price breaks above Donchian high AND ADX > 25 AND volume confirmed
            if curr_close > donchian_high_aligned[i] and strong_trend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND ADX > 25 AND volume confirmed
            elif curr_close < donchian_low_aligned[i] and strong_trend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Donchian low
            if curr_close < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Donchian high
            if curr_close > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1wADX_Regime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0