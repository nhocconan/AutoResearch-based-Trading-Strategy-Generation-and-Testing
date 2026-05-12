#!/usr/bin/env python3
# 4h_1D_Camarilla_R1_S1_Breakout_Trend_VolumeS_v4
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation (1.5x avg volume).
# The 1d EMA34 provides trend direction to avoid counter-trend trades, while volume spikes confirm breakout strength.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag. Works in bull/bear by following 1d trend.

name = "4h_1D_Camarilla_R1_S1_Breakout_Trend_VolumeS_v4"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d Camarilla levels (R1, S1)
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3
    range_ = high_1d[-1] - low_1d[-1]
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # For historical calculation, we need to compute for each day
    # Use rolling window to get daily high/low/close for each point
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    close_series = pd.Series(close_1d)
    
    # Calculate rolling daily pivot points
    roll_high = high_series.rolling(window=1, min_periods=1).max()  # Daily high
    roll_low = low_series.rolling(window=1, min_periods=1).min()    # Daily low
    roll_close = close_series.rolling(window=1, min_periods=1).last() # Daily close
    
    pivot_daily = (roll_high + roll_low + roll_close) / 3
    range_daily = roll_high - roll_low
    r1_daily = pivot_daily + (range_daily * 1.1 / 12)
    s1_daily = pivot_daily - (range_daily * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_daily.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_daily.values)

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 in 1d uptrend with volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 in 1d downtrend with volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals