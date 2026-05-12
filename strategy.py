#!/usr/bin/env python3
"""
6h_Bollinger_Bandwidth_Expansion_1dTrend_Filter
Hypothesis: Bollinger Bandwidth expansion from low volatility indicates the start of a trending move. Combined with 1d EMA50 trend filter and volume confirmation, this captures explosive moves in both bull and bear markets while avoiding range-bound chop. Uses 6h timeframe with 1d EMA50 trend filter for higher timeframe context.
"""

name = "6h_Bollinger_Bandwidth_Expansion_1dTrend_Filter"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Bollinger Bands (20-period SMA, 2 std dev)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_dev = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_dev
    bb_lower = sma_20 - 2 * std_dev
    bb_width = bb_upper - bb_lower

    # Bollinger Bandwidth percentile (20-period lookback) - low when < 20th percentile
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    # Handle NaN from rolling apply
    bb_width_percentile = np.where(np.isnan(bb_width_percentile), 50, bb_width_percentile)

    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 and Bollinger warmup
        if (np.isnan(sma_20[i]) or np.isnan(std_dev[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bollinger Bandwidth expanding from low volatility (<20th percentile) + 
            # Price above SMA20 + 1d Uptrend + Volume confirmation
            if (bb_width_percentile[i] < 20 and  # Low volatility squeeze
                close[i] > sma_20[i] and         # Price above SMA20
                close[i] > ema_50_1d_aligned[i] and  # 1d Uptrend
                volume_confirm[i]):              # Volume confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: Bollinger Bandwidth expanding from low volatility (<20th percentile) + 
            # Price below SMA20 + 1d Downtrend + Volume confirmation
            elif (bb_width_percentile[i] < 20 and   # Low volatility squeeze
                  close[i] < sma_20[i] and          # Price below SMA20
                  close[i] < ema_50_1d_aligned[i] and  # 1d Downtrend
                  volume_confirm[i]):               # Volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below SMA20 OR volatility contraction (bandwidth > 80th percentile)
            if close[i] < sma_20[i] or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above SMA20 OR volatility contraction (bandwidth > 80th percentile)
            if close[i] > sma_20[i] or bb_width_percentile[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals