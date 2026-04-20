#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise - in trending markets it follows price closely,
# in ranging markets it stays flat. Combined with volume confirmation (>1.5x 20-period average) and ADX>25 for trend strength,
# this captures strong trending moves while avoiding whipsaws in ranging markets. Works in both bull and bear markets
# by following the dominant trend. Position size 0.25 to manage risk. Target: 20-40 trades/year (80-160 total over 4 years).

name = "4h_KAMA_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for faster trend confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (2-period efficiency, 30-period fast, 30-period slow)
    # ER = Efficiency Ratio = abs(close - close[30]) / sum(abs(close - close[1])) over 30 periods
    change = np.abs(np.diff(close, n=30))  # abs(close[i] - close[i-30])
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of abs changes over window
    
    # Pad arrays for alignment
    change_padded = np.full(n, np.nan)
    volatility_padded = np.full(n, np.nan)
    change_padded[30:] = change
    volatility_padded[30:] = volatility
    
    # Calculate ER with proper handling
    er = np.full(n, np.nan)
    valid_vol = volatility_padded > 0
    er[valid_vol] = change_padded[valid_vol] / volatility_padded[valid_vol]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for 2-period EMA
    slow_sc = 2 / (30 + 1)  # for 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get 1h data for confirmation
    close_1h = df_1h['close'].values
    # Calculate 21-period EMA on 1h
    ema_21_1h = pd.Series(close_1h).ewm(span=21, adjust=False).mean().values
    ema_21_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_21_1h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    # ADX filter for trend strength (using 1h data for responsiveness)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = high_1h[1:] - low_1h[1:]
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1h[1:] - high_1h[:-1]) > (low_1h[:-1] - low_1h[1:]), 
                       np.maximum(high_1h[1:] - high_1h[:-1], 0), 0)
    dm_minus = np.where((low_1h[:-1] - low_1h[1:]) > (high_1h[1:] - high_1h[:-1]), 
                        np.maximum(low_1h[:-1] - low_1h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM (14-period)
    tr_sum = np.full_like(high_1h, np.nan)
    dm_plus_sum = np.full_like(high_1h, np.nan)
    dm_minus_sum = np.full_like(high_1h, np.nan)
    
    for i in range(len(high_1h)):
        if i >= 13:  # 14-period
            tr_sum[i] = np.nansum(tr[i-13:i+1])
            dm_plus_sum[i] = np.nansum(dm_plus[i-13:i+1])
            dm_minus_sum[i] = np.nansum(dm_minus[i-13:i+1])
    
    # Directional Indicators
    di_plus = np.full_like(high_1h, np.nan)
    di_minus = np.full_like(high_1h, np.nan)
    dx = np.full_like(high_1h, np.nan)
    
    valid = ~np.isnan(tr_sum) & (tr_sum != 0)
    di_plus[valid] = 100 * dm_plus_sum[valid] / tr_sum[valid]
    di_minus[valid] = 100 * dm_minus_sum[valid] / tr_sum[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX (smoothed DX)
    adx_1h = np.full_like(high_1h, np.nan)
    for i in range(len(high_1h)):
        if i >= 27:  # 14 + 13 for ADX
            valid_dx = dx[i-13:i+1]
            if not np.all(np.isnan(valid_dx)):
                adx_1h[i] = np.nanmean(valid_dx)
    
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 21)  # Ensure KAMA and 1h indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_21_1h_aligned[i]) or 
            np.isnan(adx_1h_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA AND price above 1h EMA21 AND ADX > 25 AND volume confirmation
            if (close[i] > kama[i] and close[i] > ema_21_1h_aligned[i] and 
                adx_1h_aligned[i] > 25 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND price below 1h EMA21 AND ADX > 25 AND volume confirmation
            elif (close[i] < kama[i] and close[i] < ema_21_1h_aligned[i] and 
                  adx_1h_aligned[i] > 25 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or ADX weakens significantly
            if close[i] < kama[i] or adx_1h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or ADX weakens significantly
            if close[i] > kama[i] or adx_1h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals