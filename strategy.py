#!/usr/bin/env python3
# 6h_Philips_Squeeze_Breakout_1dTrend_Volume
# Hypothesis: Philips curve-inspired squeeze breakout combining Bollinger Band width (volatility) and Donchian channel breakouts, filtered by 1d EMA trend and volume confirmation. 
# In low volatility regimes (BB width < 20th percentile), price compresses and builds energy for explosive moves. Breakouts from this squeezed state have higher follow-through.
# The 1d EMA50 filter ensures alignment with the daily trend to avoid counter-trend whipsaws. Volume confirmation adds conviction.
# Works in both bull and bear markets by capturing volatility breakouts in the direction of the higher timeframe trend.

name = "6h_Philips_Squeeze_Breakout_1dTrend_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Bollinger Bands (20, 2.0) on 6h
    bb_period = 20
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bollinger_upper = sma_20 + 2.0 * std_20
    bollinger_lower = sma_20 - 2.0 * std_20
    bb_width = (bollinger_upper - bollinger_lower) / sma_20  # Normalized width

    # Calculate BB width percentile (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values

    # Calculate Donchian Channel (20) on 6h
    dc_period = 20
    donchian_high = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    donchian_low = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: BB width in low volatility regime (< 20th percentile) + breakout above Donchian high + above 1d EMA50 + volume spike
            if (bb_width_percentile[i] < 20 and 
                close[i] > donchian_high[i] and
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: BB width in low volatility regime (< 20th percentile) + breakdown below Donchian low + below 1d EMA50 + volume spike
            elif (bb_width_percentile[i] < 20 and 
                  close[i] < donchian_low[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low or volatility expands (BB width > 80th percentile)
            if (close[i] < donchian_low[i] or bb_width_percentile[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high or volatility expands (BB width > 80th percentile)
            if (close[i] > donchian_high[i] or bb_width_percentile[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals