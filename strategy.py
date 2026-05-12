#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout on 12h with 1-day EMA34 trend filter and volume spike confirmation.
# Camarilla levels identify key support/resistance from prior 1-day range. Breakout with trend and volume
# provides high-probability entries. Designed for low trade frequency (target 12-37/year) to minimize fee drag.
# Works in bull/bear markets by following daily trend direction. Exit on opposite Camarilla level touch.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla calculation (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from prior 1-day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's range
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = C + 0.115*range, S1 = C - 0.115*range
    camarilla_r1 = close_1d + 0.115 * range_1d
    camarilla_s1 = close_1d - 0.115 * range_1d
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Get 1d data for EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate volume spike threshold (2.0x 10-period SMA on 12h)
    volume_series = pd.Series(volume)
    volume_sma10 = volume_series.rolling(window=10, min_periods=10).mean().values
    volume_spike_threshold = volume_sma10 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to access previous bar
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 in uptrend with volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i-1] <= camarilla_r1_aligned[i] and  # Confirm breakout
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_sma10[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 in downtrend with volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i-1] >= camarilla_s1_aligned[i] and  # Confirm breakdown
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_sma10[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or crosses Camarilla S1 (mean reversion)
            if close[i] <= camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or crosses Camarilla R1 (mean reversion)
            if close[i] >= camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals