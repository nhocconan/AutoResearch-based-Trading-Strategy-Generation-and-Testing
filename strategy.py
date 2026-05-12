#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1dTrend_VolumeFilter
Hypothesis: In BTC/ETH, price breaking above/below the 20-period Keltner Channel (ATR-based) 
with 1-day EMA trend alignment and volume expansion (>1.5x 20-period average) captures 
momentum with low turnover. Uses ADX > 20 to filter range conditions. Targets 20-30 trades/year.
"""

name = "4h_Keltner_Breakout_1dTrend_VolumeFilter"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 1d ATR(14) for Keltner Channel
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Calculate 1d EMA20 for Keltner middle line
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20_1d + 2 * atr14
    lower_keltner = ema20_1d - 2 * atr14
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)

    # Calculate 1d ADX(14) for trend filter
    # +DM and -DM
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Shift to align with original array
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 4h bar
        ema50 = ema50_1d_aligned[i]
        upper_kelt = upper_keltner_aligned[i]
        lower_kelt = lower_keltner_aligned[i]
        adx_val = adx_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(upper_kelt) or 
            np.isnan(lower_kelt) or np.isnan(adx_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: only trade when ADX > 20 (trending market)
        if adx_val <= 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Keltner + price above EMA50 + volume surge
            if (close[i] > upper_kelt and 
                close[i] > ema50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner + price below EMA50 + volume surge
            elif (close[i] < lower_kelt and 
                  close[i] < ema50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Keltner or price below EMA50
            if (close[i] < lower_kelt or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Keltner or price above EMA50
            if (close[i] > upper_kelt or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals