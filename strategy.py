#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol
# Hypothesis: 1h price breaks daily Camarilla R1/S1 levels with 4h trend filter (EMA20) and 1d volume spike confirmation.
# Uses daily pivots for key support/resistance, 4h EMA20 for trend direction (avoids counter-trend trades), and 1d volume spike (2x avg) to confirm breakout strength.
# Designed for 15-30 trades/year on 1h timeframe by requiring multiple confluence factors.
# Works in bull/bear markets by following 4h trend direction - only takes longs in 4h uptrend, shorts in 4h downtrend.
# Exits when price crosses 4h EMA20 (trend reversal) to avoid giving back profits.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
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

    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)

    close_4h = df_4h['close'].values

    # Calculate daily volume SMA20 for volume confirmation
    volume_1d_series = pd.Series(volume_1d)
    volume_sma20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20_1d * 2.0  # Require 2x average daily volume

    # Calculate 4h EMA20 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate daily Camarilla pivot levels: R1, S1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0

    # Align HTF data to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    volume_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_sma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 in 4h uptrend with volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema20_4h_aligned[i] and 
                volume_1d[-1] > volume_sma20_1d_aligned[i]):  # Use latest daily volume
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below S1 in 4h downtrend with volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema20_4h_aligned[i] and 
                  volume_1d[-1] > volume_sma20_1d_aligned[i]):  # Use latest daily volume
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 4h EMA20 (trend reversal)
            if close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above 4h EMA20 (trend reversal)
            if close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals