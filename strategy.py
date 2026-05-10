#!/usr/bin/env python3
# 4h_KAMA_Trend_Adaptive_Exit
# Hypothesis: KAMA adapts to market noise, making it ideal for BTC/ETH trend following in both bull and bear markets.
# Uses KAMA crossover for entries, with adaptive exit based on trend strength (ADX) to avoid whipsaws.
# Target: 20-30 trades/year to minimize fee drag while capturing major trends.

name = "4h_KAMA_Trend_Adaptive_Exit"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h KAMA (ER=10)
    close = prices['close'].values
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 10)))
    volatility = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(tr, np.nan, dtype=float)
    for i in range(1, len(tr)):
        if np.isnan(tr[i-1]):
            atr[i] = np.nanmean(tr[max(0, i-13):i+1]) if i >= 14 else np.nan
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14 if i >= 14 else np.nan
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = np.full_like(high_1d, np.nan, dtype=float)
    minus_dm_smooth = np.full_like(high_1d, np.nan, dtype=float)
    for i in range(1, len(high_1d)):
        if i == 1:
            plus_dm_smooth[i] = np.nansum(plus_dm[max(0, i-13):i+1]) if i >= 13 else np.nan
            minus_dm_smooth[i] = np.nansum(minus_dm[max(0, i-13):i+1]) if i >= 13 else np.nan
        else:
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/14) + plus_dm[i-1]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/14) + minus_dm[i-1]
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX
    adx = np.full_like(dx, np.nan, dtype=float)
    for i in range(len(dx)):
        if i < 27:
            adx[i] = np.nan
        elif i == 27:
            adx[i] = np.nanmean(dx[14:28])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period EMA
    volume = prices['volume'].values
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10) and ADX (27)
    start_idx = 27
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with strengthening trend (ADX > 25) and volume
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with strengthening trend (ADX > 25) and volume
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakening (ADX < 20) or price crosses below KAMA
            if adx_aligned[i] < 20 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakening (ADX < 20) or price crosses above KAMA
            if adx_aligned[i] < 20 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals