#!/usr/bin/env python3
# 6h_ADX_Stochastic_Reversal_1dTrend_Volume
# Hypothesis: Mean-reversion pullbacks in strong trends using ADX (trend strength) + Stochastic (overbought/oversold).
# Enter long when ADX > 25 (strong trend) and %K crosses above 20 from below (oversold bounce).
# Enter short when ADX > 25 and %K crosses below 80 from above (overbought rejection).
# Exit when Stochastic returns to neutral range (40-60) or ADX weakens (<20).
# Trend filter from 1d EMA50 ensures alignment with higher timeframe momentum.
# Works in bull/bear by fading overextensions within the dominant trend.
# Target: 20-35 trades/year to minimize fee drag.

name = "6h_ADX_Stochastic_Reversal_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    # Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr

    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values

    # Stochastic Oscillator (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    # Avoid division by zero
    k_percent = np.where((highest_high - lowest_low) == 0, 50, k_percent)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values

    # Volume confirmation: volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(adx[i]) or np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Strong trend (ADX>25) + Stochastic crosses above 20 from below + price > 1d EMA50 + volume
            if (adx[i] > 25 and 
                k_percent[i] > 20 and d_percent[i-1] <= 20 and  # cross above 20
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: Strong trend (ADX>25) + Stochastic crosses below 80 from above + price < 1d EMA50 + volume
            elif (adx[i] > 25 and 
                  k_percent[i] < 80 and d_percent[i-1] >= 80 and  # cross below 80
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic returns to neutral (>40) or trend weakens (ADX<20)
            if k_percent[i] > 40 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Stochastic returns to neutral (<60) or trend weakens (ADX<20)
            if k_percent[i] < 60 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals