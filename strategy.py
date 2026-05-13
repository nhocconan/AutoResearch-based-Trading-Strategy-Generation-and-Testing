#!/usr/bin/env python3
# 1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
# Hypothesis: 1D Camarilla R3/S3 breakout with volume confirmation and weekly trend filter.
# Long when price breaks above R3 with volume spike and weekly close above weekly open (bullish week).
# Short when price breaks below S3 with volume spike and weekly close below weekly open (bearish week).
# Exit when price returns to Pivot (central level) or opposite Camarilla level.
# Designed for 10-25 trades/year to minimize fee drag. Works in bull/bear via weekly trend alignment.

name = "1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Camarilla levels (based on previous day's OHLC)
    R3 = np.full(n, np.nan)
    S3 = np.full(n, np.nan)
    Pivot = np.full(n, np.nan)
    R4 = np.full(n, np.nan)
    S4 = np.full(n, np.nan)

    for i in range(1, n):
        # Calculate based on previous day's range
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        Pivot[i] = (prev_high + prev_low + prev_close) / 3
        R3[i] = Pivot[i] + range_val * 1.1 / 2
        S3[i] = Pivot[i] - range_val * 1.1 / 2
        R4[i] = Pivot[i] + range_val * 1.1
        S4[i] = Pivot[i] - range_val * 1.1

    # Volume confirmation: current volume > 2.0 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Weekly trend: weekly close > weekly open (bullish) or < (bearish)
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(Pivot[i]) or np.isnan(volume_spike[i]) or np.isnan(weekly_bullish_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R3 with volume spike and bullish weekly trend
            if close[i] > R3[i] and volume_spike[i] and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with volume spike and bearish weekly trend
            elif close[i] < S3[i] and volume_spike[i] and weekly_bullish_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to Pivot or drops below S3 (reversal signal)
            if close[i] <= Pivot[i] or close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to Pivot or rises above R3 (reversal signal)
            if close[i] >= Pivot[i] or close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals