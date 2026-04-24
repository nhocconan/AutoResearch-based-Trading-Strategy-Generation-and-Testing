#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d ADX trend filter.
- Uses Donchian channel (20-bar high/low) from prior completed 6h candles to identify breakouts.
- Volume confirmation: current volume > 1.5x 20-bar average on 6b timeframe.
- Trend filter: 1d ADX > 25 to ensure trending market (avoid whipsaws in ranging markets).
- Designed for 6h timeframe to capture medium-term breakouts with lower frequency than 4h.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
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
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no prior close
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    plus_di = 100 * wilders_smoothing(plus_dm, period_adx) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period_adx) / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = wilders_smoothing(dx, period_adx)
    
    # Get 6h data ONCE before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need for Donchian calculation
        return np.zeros(n)
    
    # Prior completed 6h OHLC for Donchian channels (shift by 1 to avoid look-ahead)
    high_6h = df_6h['high'].shift(1).values
    low_6h = df_6h['low'].shift(1).values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_6h, 20)
    donchian_low = rolling_min(low_6h, 20)
    
    # Align HTF indicators to LTF (prices timeframe)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Volume confirmation: > 1.5x 20-period average on LTF
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 34, 20)  # ADX needs ~34 bars (14+14+6), Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ADX trend filter (> 25 indicates trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: breakout above Donchian high AND volume confirmation AND trending market
            if close[i] > donchian_high_aligned[i] and volume_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND volume confirmation AND trending market
            elif close[i] < donchian_low_aligned[i] and volume_confirm and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low OR loss of trend
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high OR loss of trend
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hVolume_1dADX_v1"
timeframe = "6h"
leverage = 1.0