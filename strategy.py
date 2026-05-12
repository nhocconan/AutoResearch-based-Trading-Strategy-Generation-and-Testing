#!/usr/bin/env python3
# 4h_CCI_MeanReversion_WithTrendFilter
# Hypothesis: CCI identifies overbought/oversold conditions for mean reversion.
# In ranging markets (CHOP > 61.8), extreme CCI readings (> +100 or < -100) offer high-probability reversals.
# A daily EMA50 filter ensures we only trade in the direction of the higher timeframe trend, avoiding counter-trend trades in strong trends.
# Designed for 4h timeframe with tight entry conditions to target 20-50 trades per year, minimizing fee drag.

name = "4h_CCI_MeanReversion_WithTrendFilter"
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

    # Get daily data for EMA trend filter and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate EMA50 on daily for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 14-period CCI
    tp = (high + low + close) / 3.0  # Typical Price
    sma_tp = pd.Series(tp).rolling(window=14, min_periods=14).mean().values
    mad = pd.Series(tp).rolling(window=14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (tp - sma_tp) / (0.015 * mad)

    # Calculate Chopiness Index (CHOP) on daily for regime filter
    # ATR14
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max high and min low over 14 periods
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    range14 = max_high14 - min_low14
    # Avoid division by zero or log of zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * (np.log10(sum_atr14) - np.log10(range14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Only trade in ranging markets (CHOP > 61.8 indicates ranging)
            if chop_aligned[i] > 61.8:
                # LONG: CCI < -100 (oversold) and price above daily EMA50 (uptrend bias)
                if cci[i] < -100 and close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: CCI > +100 (overbought) and price below daily EMA50 (downtrend bias)
                elif cci[i] > 100 and close[i] < ema50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Trending market, stay flat
        elif position == 1:
            # EXIT LONG: CCI returns to neutral (> -50) or trend alignment breaks
            if cci[i] > -50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI returns to neutral (< +50) or trend alignment breaks
            if cci[i] < 50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals