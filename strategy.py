#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot breakouts (R3/S3) from daily levels with 1d EMA trend filter and volume confirmation.
# Works in bull markets by capturing breakouts and in bear markets by fading mean reversion at extreme levels.
# Volume filter reduces false signals. Designed for ~15-25 trades/year on 12h timeframe to minimize fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L) * 1.1/2, S3 = C - (H-L) * 1.1/2
    # where C, H, L are from previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle first value
    prev_close = np.where(np.isnan(prev_close), close[0], prev_close)
    prev_high = np.where(np.isnan(prev_high), high[0], prev_high)
    prev_low = np.where(np.isnan(prev_low), low[0], prev_low)
    
    camarilla_upper = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_lower = prev_close - (prev_high - prev_low) * 1.1 / 2

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 2.0x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    # Align Camarilla levels to 12h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1d, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1d, camarilla_lower)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 needs 34 bars
        # Skip if any required data is NaN
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 + volume + 1d uptrend
            if (close[i] > camarilla_upper_aligned[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + volume + 1d downtrend
            elif (close[i] < camarilla_lower_aligned[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla H4 level OR 1d trend turns down
            # H4 = C + (H-L) * 1.1/6
            camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 6
            camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            if (close[i] < camarilla_h4_aligned[i] or
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla L4 level OR 1d trend turns up
            # L4 = C - (H-L) * 1.1/6
            camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 6
            camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            if (close[i] > camarilla_l4_aligned[i] or
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals