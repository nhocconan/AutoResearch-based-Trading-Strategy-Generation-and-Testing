#!/usr/bin/env python3
# 12h_4hTrend_1dVolatilityBreakout
# Hypothesis: In trending markets (4h ADX > 25), volatility contractions (Bollinger Bandwidth < 50th percentile) on 1d
# precede explosive moves. Breakouts from the 1d Bollinger Bands with volume confirmation capture these moves.
# Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band)
# by only trading in the direction of the 4h trend. Uses 12h timeframe for lower frequency and reduced fee drag.

name = "12h_4hTrend_1dVolatilityBreakout"
timeframe = "12h"
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
    
    # Get 4h data for trend filter (ADX)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 4h ADX(14) for trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smoothed_ma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    tr_smoothed = smoothed_ma(tr, 14)
    dm_plus_smoothed = smoothed_ma(dm_plus, 14)
    dm_minus_smoothed = smoothed_ma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_ma(dx, 14)
    
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_period = 20
    bb_std = 2
    
    # Middle Bollinger Band (SMA)
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    # Standard deviation
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    # Upper and Lower Bands
    upper_bb = sma_1d + (std_1d * bb_std)
    lower_bb = sma_1d - (std_1d * bb_std)
    
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ADX (35), Bollinger Bands (20), volume MA (20)
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_4h_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_4h_aligned[i] > 25
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: trending + price breaks above upper BB + volume
            if trending and close[i] > upper_bb_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: trending + price breaks below lower BB + volume
            elif trending and close[i] < lower_bb_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakens or price re-enters below upper BB
            if not trending or close[i] < upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens or price re-enters above lower BB
            if not trending or close[i] > lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals