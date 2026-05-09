#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day KAMA trend and 12h volume confirmation.
# In trending markets (KAMA slope > 0), enter long on volume breakout above 20-period average.
# In ranging markets (KAMA slope < 0), enter short on volume breakout below 20-period average.
# Uses 1-week ADX to filter weak trends (ADX < 20) and avoid whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_KAMA_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day KAMA for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    # Efficiency ratio
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    kama = pd.Series(kama, index=close_1d.index)
    kama_slope = kama.diff()
    kama_slope_values = kama_slope.values
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_slope_values)
    
    # 1-week ADX for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = np.where((high_1w - high_1w.shift(1)) > (low_1w.shift(1) - low_1w), 
                       np.maximum(high_1w - high_1w.shift(1), 0), 0)
    dm_minus = np.where((low_1w.shift(1) - low_1w) > (high_1w - high_1w.shift(1)), 
                        np.maximum(low_1w.shift(1) - low_1w, 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, 1e-10)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # 12h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_ma_values = volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_slope_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade when trend is strong enough (ADX >= 20)
            if adx_aligned[i] >= 20:
                # Trending up: KAMA slope positive + volume above average
                if kama_slope_aligned[i] > 0 and volume[i] > volume_ma_values[i]:
                    signals[i] = 0.25
                    position = 1
                # Trending down: KAMA slope negative + volume above average
                elif kama_slope_aligned[i] < 0 and volume[i] > volume_ma_values[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: trend weakens (ADX < 20) or KAMA slope turns negative
            if adx_aligned[i] < 20 or kama_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakens (ADX < 20) or KAMA slope turns positive
            if adx_aligned[i] < 20 or kama_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals