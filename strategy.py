#!/usr/bin/env python3
# 4h_ADX_KAMA_Trend_1dTrend_Volume
# Hypothesis: KAMA trend direction filtered by daily ADX and volume capture sustainable trends while avoiding chop. 
# In bull markets, KAMA upward + ADX strong + volume confirms uptrend. In bear markets, KAMA downward + ADX strong + volume confirms downtrend.
# Uses ADX to filter regime (trending vs ranging) and KAMA for adaptive trend following. 
# Target: 20-40 trades per year per symbol to minimize fee drag.

name = "4h_ADX_KAMA_Trend_1dTrend_Volume"
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

    # Get 1d data for ADX and KAMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX (14) on daily timeframe
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = max(high - prev_high, 0) if > max(prev_low - low, 0) else 0
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    # -DM = max(prev_low - low, 0) if > max(high - prev_high, 0) else 0
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # SC for fastest EMA
    slow_sc = 2 / (30 + 1) # SC for slowest EMA
    
    # Efficiency Ratio
    change = np.abs(df_1d['close'] - df_1d['close'].shift(er_length))
    volatility = np.abs(df_1d['close'] - df_1d['close'].shift(1)).rolling(window=er_length, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing Constant
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.full_like(df_1d['close'], np.nan, dtype=float)
    kama[er_length] = df_1d['close'].iloc[er_length]  # Initialize
    for i in range(er_length + 1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Align ADX and KAMA to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA uptrend + ADX strong (>25) + volume spike
            if (close[i] > kama_aligned[i] and 
                adx_aligned[i] > 25 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA downtrend + ADX strong (>25) + volume spike
            elif (close[i] < kama_aligned[i] and 
                  adx_aligned[i] > 25 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA downtrend or ADX weak (<20)
            if close[i] < kama_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA uptrend or ADX weak (<20)
            if close[i] > kama_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals