#!/usr/bin/env python3
"""
12h_KAMA_Trend_VolumeSpike_1dTrend
Hypothesis: 12h KAMA trend direction filtered by 1d ADX trend (>25) and volume spike (>2x 20-day average).
KAMA adapts to market noise, reducing whipsaws in choppy markets. Volume spike confirms institutional interest.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
"""

name = "12h_KAMA_Trend_VolumeSpike_1dTrend"
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
    
    # Get 1d data for filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (2, 10, 30) - ER smoothing, 10 fast, 30 slow
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Pad volatility to match length
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded != 0, change / volatility_padded, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA trend direction
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    kama_up[0] = False
    kama_down[0] = False
    
    # 1d ADX for trend filter (14-period)
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
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder smoothing)
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr_period = 14
    atr = smooth_wilder(tr, atr_period)
    dm_plus_smooth = smooth_wilder(dm_plus, atr_period)
    dm_minus_smooth = smooth_wilder(dm_minus, atr_period)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, atr_period)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Align 1d indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is already 12h, but align for safety
    
    # Volume spike condition: current 12h volume > 2x 20-period average of 12h volume
    vol_ma_12h = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_12h[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # Prevent overtrading (48 hours)
    
    start_idx = max(20, 30)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_12h[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction using ADX and price vs 20-period SMA
        sma_20_1d = np.full_like(close_1d, np.nan)
        for j in range(20, len(close_1d)):
            sma_20_1d[j] = np.mean(close_1d[j-20:j])
        sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
        
        if not np.isnan(sma_20_1d_aligned[i]):
            trend_1d_up = adx_aligned[i] > 25 and close_1d_aligned[i] > sma_20_1d_aligned[i]
            trend_1d_down = adx_aligned[i] > 25 and close_1d_aligned[i] < sma_20_1d_aligned[i]
        else:
            trend_1d_up = False
            trend_1d_down = False
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: KAMA upward in 1d uptrend with volume spike
            if (kama_up[i] and 
                trend_1d_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: KAMA downward in 1d downtrend with volume spike
            elif (kama_down[i] and 
                  trend_1d_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: KAMA turns downward
            if kama_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns upward
            if kama_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals