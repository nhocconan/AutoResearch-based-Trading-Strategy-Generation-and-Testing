#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX(25) trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves in both bull and bear markets.
# 1d ADX > 25 ensures we only trade in strong trending regimes, avoiding whipsaws.
# Volume confirmation (1.5x 20-period EMA) filters low-conviction breakouts.
# Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year.

name = "4h_Donchian20_1dADX25_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    def wilders_smooth_dx(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # Find first valid value for seeding
        valid_idx = np.where(~np.isnan(data))[0]
        if len(valid_idx) < period:
            return result
        start_idx = valid_idx[0] + period - 1
        if start_idx >= len(data):
            return result
        # First value: simple average of valid data
        result[start_idx] = np.nanmean(data[valid_idx[0]:start_idx+1])
        # Subsequent values
        for i in range(start_idx + 1, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    adx_14 = wilders_smooth_dx(dx, 14)
    adx_25_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Donchian channels (20-period) from previous bar
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip invalid data or outside session
        if (np.isnan(adx_25_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: > 1.5x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Strong trend: ADX > 25
        strong_trend = adx_25_aligned[i] > 25
        
        if position == 0:
            # Long entry: break above upper Donchian in strong uptrend with volume
            if close[i] > donchian_upper[i] and strong_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower Donchian in strong downtrend with volume
            elif close[i] < donchian_lower[i] and strong_trend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below lower Donchian or loss of strong trend
            if close[i] < donchian_lower[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above upper Donchian or loss of strong trend
            if close[i] > donchian_upper[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals