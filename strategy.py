#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX (triple EMA) momentum confirms trend when crossing zero. 
Long when TRIX crosses above zero with volume spike (>2x 20-bar average) and 
choppiness regime indicates trend (CHOP < 38.2). Short when TRIX crosses below zero 
with volume spike and CHOP < 38.2. Uses 1d trend filter (price > EMA50) for long 
bias and (price < EMA50) for short bias to avoid counter-trend trades. 
Designed for 4h timeframe to target 20-50 trades/year with low turnover.
Works in bull via momentum continuation and bear via counter-trend bounces at extremes.
"""

name = "4h_TRIX_VolumeSpike_Regime"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate TRIX (15-period triple EMA) on close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    trix = trix.fillna(0).values

    # Calculate 40-period Choppiness Index
    # CHOP = 100 * log15(sum(ATR(1)) / (max(high) - min(low))) / log15(period)
    tr1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values
    sum_tr1 = pd.Series(atr1).rolling(window=40, min_periods=40).sum().values
    max_high = pd.Series(high).rolling(window=40, min_periods=40).max().values
    min_low = pd.Series(low).rolling(window=40, min_periods=40).min().values
    chop = 100 * (np.log10(sum_tr1) - np.log10(max_high - min_low)) / np.log10(40)
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # Avoid division by zero
    chop = np.nan_to_num(chop, nan=50.0)

    # Volume confirmation: 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Get aligned values for current 4h bar
        ema50 = ema50_1d_aligned[i]
        chop_val = chop[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(chop_val) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: only trade when CHOP < 38.2 (trending market)
        if chop_val >= 38.2:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + price above 1d EMA50
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                volume[i] > vol_avg_val * 2.0 and 
                close[i] > ema50):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + price below 1d EMA50
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  volume[i] > vol_avg_val * 2.0 and 
                  close[i] < ema50):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or price below 1d EMA50
            if (trix[i] < 0 and trix[i-1] >= 0) or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or price above 1d EMA50
            if (trix[i] > 0 and trix[i-1] <= 0) or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals