#!/usr/bin/env python3
# 12h_MedianReversion_VolumeSpike
# Hypothesis: Mean reversion at 12h extremes using median price vs SMA deviation, with volume spike confirmation and 1d trend filter.
# Works in both bull/bear markets: in uptrend, buy dips below median; in downtrend, sell rallies above median.
# Median is more robust to outliers than mean. Volume spike confirms institutional interest at extremes.
# Trend filter ensures we trade with higher timeframe momentum to avoid counter-trend whipsaws.
# Targets low trade frequency (<50/year) with high conviction entries.

name = "12h_MedianReversion_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter and median calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate 1d median price (more robust than mean)
    median_price_1d = np.nanmedian(np.column_stack([high_1d, low_1d, close_1d]), axis=1)
    
    # Calculate 1d SMA(50) for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate price deviation from median as percentage
    deviation_pct = (close_1d - median_price_1d) / median_price_1d * 100
    
    # Calculate volume spike threshold (2.5x 20-period SMA on 12h)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.5

    # Align 1d indicators to 12h timeframe
    median_price_1d_aligned = align_htf_to_ltf(prices, df_1d, median_price_1d)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    deviation_pct_aligned = align_htf_to_ltf(prices, df_1d, deviation_pct)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(median_price_1d_aligned[i]) or np.isnan(sma50_1d_aligned[i]) or 
            np.isnan(deviation_pct_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price below median (-1.5% deviation) in uptrend with volume spike
            if (deviation_pct_aligned[i] < -1.5 and 
                close[i] > sma50_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above median (+1.5% deviation) in downtrend with volume spike
            elif (deviation_pct_aligned[i] > 1.5 and 
                  close[i] < sma50_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to median or trend breaks
            if (deviation_pct_aligned[i] > -0.5 or close[i] < sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to median or trend breaks
            if (deviation_pct_aligned[i] < 0.5 or close[i] > sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals