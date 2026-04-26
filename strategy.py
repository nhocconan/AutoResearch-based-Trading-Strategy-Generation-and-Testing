#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeRegime
Hypothesis: 4h Camarilla R3/S3 breakout with 1w trend filter (price > EMA50) and volume confirmation (1.8x EMA20 volume), only in trending regimes (ADX > 25 on 1d).
Enters long when price breaks above R3 with bullish 1w trend, volume spike, and trending regime.
Enters short when price breaks below S3 with bearish 1w trend, volume spike, and trending regime.
Exits when price reverts to opposite Camarilla level (S3 for longs, R3 for shorts).
Designed for 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets by following 1w trend and regime filter.
"""

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
    
    # Calculate Camarilla pivot levels (R3, S3) on 1d timeframe
    # Use prior completed 1d bar to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1d bar's OHLC for Camarilla calculation (shifted by 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_open = np.roll(df_1d['open'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    prior_open[0] = np.nan
    
    # Calculate pivot point and Camarilla levels (R3, S3)
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_hl = prior_high - prior_low
    r3 = pivot + range_hl * 1.1 / 2.0
    s3 = pivot - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume)
    
    # Regime filter: ADX > 25 on 1d timeframe (trending market)
    # Calculate ADX components: +DI, -DI, DX
    period = 14
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # first value has no prior close
    
    # +DM and -DM
    up_move = df_1d['high'].values - np.roll(df_1d['high'].values, 1)
    down_move = np.roll(df_1d['low'].values, 1) - df_1d['low'].values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr = wilders_smoothing(tr, period)
    plus_di_smoothed = wilders_smoothing(plus_dm, period)
    minus_di_smoothed = wilders_smoothing(minus_dm, period)
    
    # DI values
    plus_di = 100 * plus_di_smoothed / atr
    minus_di = 100 * minus_di_smoothed / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    def wilders_smoothing_dx(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    adx = wilders_smoothing_dx(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trending_regime = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 1d shift + 50-period EMA + ADX period)
    start_idx = 1 + max(50, period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(trending_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 + bullish 1w trend + volume spike + trending regime
        if close[i] > r3_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i] and trending_regime[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + bearish 1w trend + volume spike + trending regime
        elif close[i] < s3_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i] and trending_regime[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Camarilla level
        elif position == 1 and close[i] < s3_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0