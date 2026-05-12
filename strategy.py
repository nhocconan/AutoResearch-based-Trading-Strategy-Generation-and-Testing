#!/usr/bin/env python3
# 6h_ADX_Trend_Strength_1d_Volume
# Hypothesis: Trend-following strategy using ADX on 6h for trend strength confirmation and volume on 1d for regime filtering.
# In bull markets: ADX > 25 + price > 6h EMA20 + 1d volume > average → long
# In bear markets: ADX > 25 + price < 6h EMA20 + 1d volume > average → short
# ADX filters out choppy markets, volume ensures participation, EMA20 defines trend direction.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear by following trend direction with strength confirmation.

name = "6h_ADX_Trend_Strength_1d_Volume"
timeframe = "6h"
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

    # Get 6h data for price action, EMA, and ADX
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    volume_1d = df_1d['volume'].values

    # Calculate EMA20 on 6h for trend direction
    ema20_6h = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate ADX on 6h (trend strength)
    # True Range
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_6h - np.roll(high_6h, 1)) > (np.roll(low_6h, 1) - low_6h), 
                       np.maximum(high_6h - np.roll(high_6h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_6h, 1) - low_6h) > (high_6h - np.roll(high_6h, 1)), 
                        np.maximum(np.roll(low_6h, 1) - low_6h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values

    # Align 6h indicators to lower timeframe
    ema20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema20_6h)
    adx_aligned = align_htf_to_ltf(prices, df_6h, adx)

    # Calculate 1d volume SMA20 for regime filter
    volume_1d_series = pd.Series(volume_1d)
    volume_sma20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma20_1d)

    # Current volume for comparison
    volume_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_6h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_sma20_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Strong uptrend (ADX > 25) + price above EMA20 + above-average volume
            if (adx_aligned[i] > 25 and 
                close[i] > ema20_6h_aligned[i] and 
                volume[i] > volume_sma20[i] and
                volume[i] > volume_sma20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Strong downtrend (ADX > 25) + price below EMA20 + above-average volume
            elif (adx_aligned[i] > 25 and 
                  close[i] < ema20_6h_aligned[i] and 
                  volume[i] > volume_sma20[i] and
                  volume[i] > volume_sma20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakness (ADX < 20) or price crosses below EMA20
            if adx_aligned[i] < 20 or close[i] < ema20_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakness (ADX < 20) or price crosses above EMA20
            if adx_aligned[i] < 20 or close[i] > ema20_6h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals