#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Volume_And_Chop
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies strong trends with low noise.
# Long when price > KAMA, volume > 1.5x SMA20, and market is trending (Choppiness Index < 38.2).
# Short when price < KAMA, volume > 1.5x SMA20, and market is trending.
# Uses 1d trend filter and volatility filter to avoid choppy periods.
# Designed for 4h timeframe to balance signal frequency and noise reduction.

name = "4h_KAMA_Trend_With_Volume_And_Chop"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate KAMA (2-period ER, 30-period smoothing)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))).reshape(-1)
    # Calculate efficiency ratio
    er = np.zeros_like(change)
    for i in range(len(change)):
        if i < 9:
            er[i] = 0
        else:
            price_change = np.abs(close[i] - close[i-9])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = price_change / (volatility_sum + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    # Choppiness Index (14-period) for regime filter
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if i < 13:
            chop[i] = 50
        else:
            atr_sum = np.sum(tr[i-13:i+1])
            hh_ll = highest_high[i] - lowest_low[i]
            chop[i] = 100 * np.log10(atr_sum / hh_ll) / np.log10(14) if hh_ll > 0 else 50

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Get aligned values for current 4h bar
        ema50_aligned = ema50_1d_aligned[i]
        chop_val = chop[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned) or np.isnan(chop_val) or 
            np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Market regime: trending when Chop < 38.2
        trending_regime = chop_val < 38.2

        if position == 0:
            # LONG: Price > KAMA + volume spike + trending regime
            if (close[i] > kama[i] and
                volume[i] > vol_threshold_val and
                trending_regime):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA + volume spike + trending regime
            elif (close[i] < kama[i] and
                  volume[i] > vol_threshold_val and
                  trending_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals