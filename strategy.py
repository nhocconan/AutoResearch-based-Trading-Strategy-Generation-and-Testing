#!/usr/bin/env python3
"""
4h_1d_camarilla_pivot_v2
Hypothesis: 4-hour strategy using daily context with Camarilla pivot levels.
Long when price crosses above daily Pivot with volume > 1.8x average and price > daily EMA200 (bullish trend).
Short when price crosses below daily Pivot with volume > 1.8x average and price < daily EMA200 (bearish trend).
Exit when price crosses opposite daily support/resistance or volume drops below average.
Uses discrete position sizing (0.25) to minimize churn. Target: 20-30 trades/year.
Added ADX filter to reduce false signals and lower trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v2"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    if len(high) < 1:
        return np.full(len(high), np.nan), np.full(len(high), np.nan)
    
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    
    return H3, L3

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    if len(high) < period + 1:
        return np.full(len(high), np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = np.full_like(tr, np.nan, dtype=float)
    dm_plus_smooth = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_smooth = np.full_like(dm_minus, np.nan, dtype=float)
    
    # Initial average
    atr[period] = np.nanmean(tr[1:period+1])
    dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
    dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
    
    # Wilder smoothing
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.full_like(tr, np.nan, dtype=float)
    di_minus = np.full_like(tr, np.nan, dtype=float)
    dx = np.full_like(tr, np.nan, dtype=float)
    
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    
    dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
    
    # ADX (smoothed DX)
    adx = np.full_like(dx, np.nan, dtype=float)
    adx[2*period] = np.nanmean(dx[period:2*period+1])
    for i in range(2*period + 1, len(dx)):
        if not np.isnan(dx[i]):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context and Pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily Pivot (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # Daily support/resistance levels
    S1_1d = pivot_1d - (range_1d * 1.1 / 12)  # Daily S1
    R1_1d = pivot_1d + (range_1d * 1.1 / 12)  # Daily R1
    
    # Calculate daily EMA for trend filter
    ema_200_1d = calculate_ema(close_1d, 200)
    
    # Calculate daily ADX for trend strength filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 4-hour timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        pivot = pivot_1d_aligned[i]
        S1 = S1_1d_aligned[i]
        R1 = R1_1d_aligned[i]
        trend_up_1d = price > ema_200_1d_aligned[i]
        strong_trend = adx_1d_aligned[i] > 25  # ADX > 25 indicates strong trend
        
        if position == 1:  # Long
            # Exit: price crosses below daily S1 or volume drops below average or trend weakens
            if price < S1 or vol_ratio < 1.0 or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above daily R1 or volume drops below average or trend weakens
            if price > R1 or vol_ratio < 1.0 or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses above daily Pivot with volume expansion and uptrend on daily
            if price > pivot and vol_ratio > 1.8 and trend_up_1d and strong_trend:
                position = 1
                signals[i] = 0.25
            # Enter short: price crosses below daily Pivot with volume expansion and downtrend on daily
            elif price < pivot and vol_ratio > 1.8 and not trend_up_1d and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals