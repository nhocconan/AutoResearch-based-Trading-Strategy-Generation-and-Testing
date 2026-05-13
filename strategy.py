#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing a robust trend filter. 
# Combines with Choppiness Index to avoid ranging markets, and uses volume 
# confirmation to ensure institutional participation. Designed to work in both 
# bull and bear markets by following the higher timeframe trend (1d) and 
# entering on pullbacks to the KAMA on 4h. Targets low-frequency, high-quality 
# setups to minimize fee drag.

name = "4h_KAMA_Trend_With_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for higher timeframe trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend direction
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    # Handle the array shapes properly
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[9] = close_1d[9]  # Start after first ER calculation
    for i in range(10, len(close_1d)):
        if np.isnan(kama_1d[i-1]):
            kama_1d[i] = close_1d[i]
        else:
            kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Calculate 1d Choppiness Index
    def true_range(high, low, close_prev):
        return np.maximum(np.maximum(high - low, np.abs(high - close_prev)), np.abs(low - close_prev))
    
    tr1d = true_range(df_1d['high'].values, df_1d['low'].values, 
                      np.concatenate([[np.nan], df_1d['close'].values[:-1]]))
    atr1d = np.full_like(tr1d, np.nan)
    for i in range(14, len(tr1d)):
        if np.isnan(atr1d[i-1]):
            atr1d[i] = np.nanmean(tr1d[i-13:i+1])
        else:
            atr1d[i] = (atr1d[i-1] * 13 + tr1d[i]) / 14
    
    maxh1d = np.full_like(df_1d['high'].values, np.nan)
    minl1d = np.full_like(df_1d['low'].values, np.nan)
    for i in range(14, len(df_1d)):
        maxh1d[i] = np.nanmax(df_1d['high'].values[i-13:i+1])
        minl1d[i] = np.nanmin(df_1d['low'].values[i-13:i+1])
    
    chop1d = 100 * np.log10(atr1d / (maxh1d - minl1d)) / np.log10(14)
    # Chop > 61.8 = ranging, Chop < 38.2 = trending
    
    # Align 1d indicators to 4h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    chop1d_aligned = align_htf_to_ltf(prices, df_1d, chop1d)

    # 4h KAMA for entry signals (more responsive)
    change_4h = np.abs(np.diff(close, n=10))
    volatility_4h = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    change_padded_4h = np.concatenate([np.full(9, np.nan), change_4h])
    volatility_padded_4h = np.concatenate([np.full(9, np.nan), volatility_4h])
    er_4h = np.where(volatility_padded_4h != 0, change_padded_4h / volatility_padded_4h, 0)
    sc_4h = (er_4h * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_4h = np.full_like(close, np.nan)
    kama_4h[9] = close[9]
    for i in range(10, len(close)):
        if np.isnan(kama_4h[i-1]):
            kama_4h[i] = close[i]
        else:
            kama_4h[i] = kama_4h[i-1] + sc_4h[i] * (close[i] - kama_4h[i-1])

    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(chop1d_aligned[i]) or
            np.isnan(kama_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade in trending markets (Chop < 38.2)
        if chop1d_aligned[i] > 38.2:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend (price above 1d KAMA) + pullback to 4h KAMA + volume spike
            if close[i] > kama_1d_aligned[i] and close[i] <= kama_4h[i] * 1.002 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price below 1d KAMA) + pullback to 4h KAMA + volume spike
            elif close[i] < kama_1d_aligned[i] and close[i] >= kama_4h[i] * 0.998 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 4h KAMA or trend turns bearish
            if close[i] < kama_4h[i] * 0.998 or close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 4h KAMA or trend turns bullish
            if close[i] > kama_4h[i] * 1.002 or close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals