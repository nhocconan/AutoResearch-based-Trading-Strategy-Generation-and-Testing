#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h ADX Trend with 4h/1d Confluence
# Hypothesis: ADX > 25 indicates strong trend on higher timeframes. 
# Enter long when 1h price pulls back to EMA21 during uptrend (4h/1d ADX>25 and 4h close>EMA50).
# Enter short when 1h price rallies to EMA21 during downtrend (4h/1d ADX>25 and 4h close<EMA50).
# Works in both bull and bear: follows the trend on higher timeframes.
# Uses 1h EMA21 for precise entry timing, reducing whipsaw.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "1h_adx_trend_4h1d_confluence_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, 14)
    
    # EMA50 on 4h
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False).mean().values
    
    # Align 4h indicators to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h EMA21 for entry timing
    close_series = pd.Series(close)
    ema21_1h = close_series.ewm(span=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema21_1h[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h and 1d
        uptrend_4h = close_4h[i//4] > ema50_4h[i//4] if i//4 < len(close_4h) else False
        uptrend_1d = close_1d[i//24] > ema50_1d[i//24] if i//24 < len(close_1d) else False
        strong_trend = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens or price breaks below EMA21
            if not (strong_trend and uptrend_4h and uptrend_1d) or close[i] < ema21_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: trend weakens or price breaks above EMA21
            if not (strong_trend and not uptrend_4h and not uptrend_1d) or close[i] > ema21_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long: uptrend on both timeframes, strong trend, price pulls back to EMA21
            if (strong_trend and uptrend_4h and uptrend_1d and 
                low[i] <= ema21_1h[i] <= high[i]):
                position = 1
                signals[i] = 0.20
            # Short: downtrend on both timeframes, strong trend, price rallies to EMA21
            elif (strong_trend and not uptrend_4h and not uptrend_1d and 
                  low[i] <= ema21_1h[i] <= high[i]):
                position = -1
                signals[i] = -0.20
    
    return signals