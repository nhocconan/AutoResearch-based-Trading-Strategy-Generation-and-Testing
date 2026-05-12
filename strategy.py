#!/usr/bin/env python3
"""
6h_Elder_Ray_Power_Trend_Filter
Hypothesis: Elder Ray (Bull Power/Bear Power) combined with 1d trend filter captures strong momentum moves.
Bull Power = High - EMA13, Bear Power = EMA13 - Low. 
Long when Bull Power > 0 and Bear Power rising (momentum building) with 1d uptrend.
Short when Bear Power < 0 and Bull Power falling with 1d downtrend.
Uses volume confirmation to avoid false signals. Designed for 50-150 trades over 4 years on 6h timeframe.
"""

name = "6h_Elder_Ray_Power_Trend_Filter"
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

    # Get 6h data for EMA13 calculation (called once before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Calculate EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_6h - ema13_6h
    bear_power = ema13_6h - low_6h

    # Align to 6h timeframe (already aligned since we used 6h data)
    bull_power_aligned = bull_power  # Already on 6h timeframe
    bear_power_aligned = bear_power  # Already on 6h timeframe

    # Get 1d data for trend filter (called once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 6h volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (strong bullish momentum) AND Bear Power decreasing (momentum building) 
            #        AND 1d uptrend AND volume confirmation
            if bull_val > 0 and bear_val < bear_power_aligned[i-1] and close[i] > ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (strong bearish momentum) AND Bull Power decreasing (momentum building)
            #        AND 1d downtrend AND volume confirmation
            elif bear_val > 0 and bull_val < bull_power_aligned[i-1] and close[i] < ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power becomes positive (bearish momentum taking over) OR 1d trend breaks
            if bear_val > 0 or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power becomes positive (bullish momentum taking over) OR 1d trend breaks
            if bull_val > 0 or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals