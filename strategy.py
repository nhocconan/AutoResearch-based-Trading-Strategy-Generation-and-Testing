#!/usr/bin/env python3
# 4h_ADX_Donchian_Breakout_1dTrend_Volume
# Hypothesis: Donchian(20) breakout on 4h combined with 1d EMA50 trend filter, volume confirmation, and ADX(14) > 25 for trend strength. 
# Works in bull markets by riding uptrends after breakouts and in bear markets by following downtrends. 
# ADX filter reduces whipsaws in ranging markets, improving signal quality and reducing trade frequency.

name = "4h_ADX_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX(14) on 4h for trend strength filter
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # avoid NaN on first element
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) >= period:
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / np.where(atr != 0, atr, 1e-10)
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / np.where(atr != 0, atr, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), ADX (14+14=28), volume MA (20)
    start_idx = max(20, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        strong_trend = adx[i] > 25
        
        # Volume confirmation (>1.5x MA)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + ADX > 25 + price breaks above Donchian high + volume
            if uptrend and strong_trend and close[i] > donchian_high[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + ADX > 25 + price breaks below Donchian low + volume
            elif downtrend and strong_trend and close[i] < donchian_low[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks, ADX weakens, or price re-enters below Donchian high
            if not uptrend or not strong_trend or close[i] < donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks, ADX weakens, or price re-enters above Donchian low
            if not downtrend or not strong_trend or close[i] > donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals