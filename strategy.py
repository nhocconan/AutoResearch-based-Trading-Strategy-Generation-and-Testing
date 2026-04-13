#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Donchian breakout for direction,
    # 1d ADX for trend strength filter, and volume spike for entry timing.
    # Long: price > 4h Donchian upper (20) AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
    # Short: price < 4h Donchian lower (20) AND 1d ADX > 25 AND volume > 1.5x 20-bar avg
    # Exit: price crosses 4h Donchian midpoint OR ADX < 20 (trend weak)
    # Session filter: 08-20 UTC to avoid low-liquidity hours.
    # Discrete sizing: 0.20 to limit drawdown and reduce fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period high/low)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high_4h = rolling_max(high_4h, 20)
    donchian_low_4h = rolling_min(low_4h, 20)
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range (TR)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        res = np.full_like(data, np.nan)
        if len(data) < period:
            return res
        # First value: simple average
        res[period-1] = np.nanmean(data[1:period])
        # Subsequent: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(res[i-1]) and not np.isnan(data[i]):
                res[i] = (res[i-1] * (period-1) + data[i]) / period
        return res
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_1d = wilders_smoothing(plus_dm, 14)
    minus_dm_1d = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di_1d = 100 * np.where(atr_1d != 0, plus_dm_1d / atr_1d, 0)
    minus_di_1d = 100 * np.where(atr_1d != 0, minus_dm_1d / atr_1d, 0)
    dx_1d = 100 * np.where((plus_di_1d + minus_di_1d) != 0, 
                           np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align 1d ADX to 1h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: >1.5x 20-bar average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in session or data not ready
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend and breakout conditions
        bullish_breakout = close[i] > donchian_high_aligned[i]
        bearish_breakout = close[i] < donchian_low_aligned[i]
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        vol_confirm = volume_spike[i]
        
        # Entry logic
        long_entry = bullish_breakout and strong_trend and vol_confirm
        short_entry = bearish_breakout and strong_trend and vol_confirm
        
        # Exit logic: breakout failure OR trend weakening
        long_exit = (close[i] < donchian_mid_aligned[i]) or weak_trend
        short_exit = (close[i] > donchian_mid_aligned[i]) or weak_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_donchian_adx_volume_v1"
timeframe = "1h"
leverage = 1.0