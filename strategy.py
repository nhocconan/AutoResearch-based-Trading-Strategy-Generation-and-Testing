#!/usr/bin/env python3
"""
6h_12h1d_adx_volume_breakout_v1
Hypothesis: 6-hour ADX trend strength + volume confirmation + daily trend filter.
Long when ADX > 25 (trending) + price breaks above 20-period high + volume > 1.5x average + daily EMA50 < EMA200 (bullish bias).
Short when ADX > 25 + price breaks below 20-period low + volume > 1.5x average + daily EMA50 > EMA200 (bearish bias).
Exit when ADX < 20 (range) or opposite breakout occurs.
Uses discrete sizing (0.25) to minimize churn. Target: 15-30 trades/year.
Works in bull/bear by requiring strong trend (ADX>25) and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_adx_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX with proper handling"""
    if len(high) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    atr = np.full_like(tr, np.nan, dtype=float)
    plus_dm_smooth = np.full_like(plus_dm, np.nan, dtype=float)
    minus_dm_smooth = np.full_like(minus_dm, np.nan, dtype=float)
    
    if len(tr) >= period:
        # Initial values
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Directional Indicators
    plus_di = np.full_like(close, np.nan, dtype=float)
    minus_di = np.full_like(close, np.nan, dtype=float)
    dx = np.full_like(close, np.nan, dtype=float)
    
    valid = ~np.isnan(atr) & (atr > 0)
    plus_di[valid] = 100 * plus_dm_smooth[valid] / atr[valid]
    minus_di[valid] = 100 * minus_dm_smooth[valid] / atr[valid]
    
    dx_valid = valid & ~np.isnan(plus_di) & ~np.isnan(minus_di) & ((plus_di + minus_di) > 0)
    dx[dx_valid] = 100 * np.abs(plus_di[dx_valid] - minus_di[dx_valid]) / (plus_di[dx_valid] + minus_di[dx_valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(close, np.nan, dtype=float)
    dx_valid_start = np.where(~np.isnan(dx))[0]
    if len(dx_valid_start) >= period:
        first_valid = dx_valid_start[0]
        adx[first_valid + period - 1] = np.nanmean(dx[first_valid:first_valid + period])
        for i in range(first_valid + period, len(adx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX and volatility context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate daily EMA50 and EMA200 for trend bias
    def calculate_ema(close_arr, period):
        if len(close_arr) < period:
            return np.full_like(close_arr, np.nan, dtype=float)
        ema = np.full_like(close_arr, np.nan, dtype=float)
        alpha = 2.0 / (period + 1)
        ema[period-1] = np.mean(close_arr[:period])
        for i in range(period, len(close_arr)):
            ema[i] = alpha * close_arr[i] + (1 - alpha) * ema[i-1]
        return ema
    
    ema_50_1d = calculate_ema(close_1d, 50)
    ema_200_1d = calculate_ema(close_1d, 200)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h price channels (20-period high/low)
    highest_20h = np.full(n, np.nan)
    lowest_20h = np.full(n, np.nan)
    for i in range(20, n):
        highest_20h[i] = np.max(high[i-20:i])
        lowest_20h[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(highest_20h[i]) or 
            np.isnan(lowest_20h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        adx = adx_12h_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        
        # Determine daily trend bias: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
        bullish_bias = ema_50 > ema_200
        bearish_bias = ema_50 < ema_200
        
        if position == 1:  # Long
            # Exit: trend weakens (ADX < 20) or price breaks below 20-period low
            if adx < 20.0 or price < lowest_20h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: trend weakens (ADX < 20) or price breaks above 20-period high
            if adx < 20.0 or price > highest_20h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: strong trend (ADX > 25) + breakout above 20-period high + volume expansion + bullish bias
            if (adx > 25.0 and price > highest_20h[i] and 
                vol_ratio > 1.5 and bullish_bias):
                position = 1
                signals[i] = 0.25
            # Enter short: strong trend (ADX > 25) + breakout below 20-period low + volume expansion + bearish bias
            elif (adx > 25.0 and price < lowest_20h[i] and 
                  vol_ratio > 1.5 and bearish_bias):
                position = -1
                signals[i] = -0.25
    
    return signals