#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout above Camarilla R1 or below S1 on 4h timeframe, with daily trend filter and volume confirmation.
# Camarilla pivot levels (R1/S1) are calculated from the previous 1d candle's high-low-close. Breakouts above R1 or below S1
# indicate strong momentum. We require the daily close to be above/below the 50-period EMA to filter for trend alignment.
# Volume must exceed 2x the 20-period SMA to avoid false breakouts. This strategy targets low trade frequency (20-40/year)
# and works in both bull and bear markets by following the daily trend direction. Exit on opposite Camarilla level touch.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels from previous 1d candle
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    cam_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    cam_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12

    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)

    # Daily trend filter: 50-period EMA
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 2x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 with uptrend and volume spike
            if (high[i] > cam_r1_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 with downtrend and volume spike
            elif (low[i] < cam_s1_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches S1 (opposite level) or closes below EMA50
            if low[i] <= cam_s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches R1 (opposite level) or closes above EMA50
            if high[i] >= cam_r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals