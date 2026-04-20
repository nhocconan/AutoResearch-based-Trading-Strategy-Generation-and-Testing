#!/usr/bin/env python3
"""
4h_KAMA_Direction_With_ADX_And_Volume_Filter
Hypothesis: Trade KAMA direction with ADX trend filter and volume confirmation.
KAMA adapts to market noise, reducing whipsaws in ranging markets. ADX ensures
trending conditions, and volume confirms institutional participation. This
combination aims to capture strong trends while avoiding false signals in chop,
working in both bull and bear markets by filtering counter-trend trades.
Target: 20-50 trades per year with position size 0.25.
"""

name = "4h_KAMA_Direction_With_ADX_And_Volume_Filter"
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
    
    # Get daily data for ADX and volume average ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate ADX (14) on daily
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_period = 14
    tr_smoothed = smooth_wilder(tr, atr_period)
    plus_dm_smoothed = smooth_wilder(plus_dm, atr_period)
    minus_dm_smoothed = smooth_wilder(minus_dm, atr_period)
    
    # DI and DX
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX
    adx = np.full_like(dx, np.nan)
    if len(dx) >= atr_period:
        adx[2*atr_period-1] = np.nanmean(dx[atr_period:2*atr_period])
        for i in range(2*atr_period, len(dx)):
            adx[i] = (adx[i-1] * (atr_period-1) + dx[i]) / atr_period
    
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Daily average volume for filter
    vol_daily = df_daily['volume'].values
    vol_ma_daily = np.full_like(vol_daily, np.nan)
    for i in range(20, len(vol_daily)):
        vol_ma_daily[i] = np.mean(vol_daily[i-20:i])
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Calculate KAMA (10, 2, 30) on 4h close
    def eratio(close, period):
        change = np.abs(np.diff(close, period))
        volatility = np.nansum(np.abs(np.diff(close)), axis=0)
        er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
        return np.concatenate([np.full(period, np.nan), er])
    
    def kama(close, er_period, fast_sc, slow_sc):
        er = eratio(close, er_period)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = np.full_like(close, np.nan)
        kama[er_period] = close[er_period]
        for i in range(er_period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10, 2/(2+1), 2/(30+1))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA + ADX > 25 + volume surge
            if close[i] > kama_vals[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA + ADX > 25 + volume surge
            elif close[i] < kama_vals[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA OR ADX < 20 (trend weakening)
            if close[i] < kama_vals[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA OR ADX < 20 (trend weakening)
            if close[i] > kama_vals[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals