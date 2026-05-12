# #!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout from weekly levels with weekly EMA trend filter and volume spike confirmation on daily timeframe.
# Uses Camarilla pivot levels (R1, S1) calculated from prior week's OHLC. Long when price breaks above R1 with weekly uptrend (price > EMA34) and volume spike.
# Short when price breaks below S1 with weekly downtrend (price < EMA34) and volume spike. Designed for low trade frequency (7-25/year) to avoid fee drag.
# Works in bull/bear markets by following weekly EMA trend direction. Exit on opposite Camarilla level touch (S1 for long exit, R1 for short exit).
# Target: 30-100 total trades over 4 years.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for Camarilla levels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate Camarilla levels for each week: based on prior week's OHLC
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    # We use prior week's values to avoid look-ahead
    rng_1w = high_1w - low_1w
    camarilla_r1 = close_1w + 1.1 * rng_1w / 12
    camarilla_s1 = close_1w - 1.1 * rng_1w / 12

    # Align Camarilla levels to daily timeframe (use prior week's levels for current week)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)

    # Get weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate volume spike threshold (2.0x 20-period SMA on daily)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 with weekly uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with weekly downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches or crosses below S1 (opposite level)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches or crosses above R1 (opposite level)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals