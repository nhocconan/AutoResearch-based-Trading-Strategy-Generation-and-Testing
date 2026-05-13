#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_Trend_Volume
# Hypothesis: Price breaking above weekly R3 (bull) or below weekly S3 (bear) with daily trend confirmation and volume capture captures institutional breakout moves.
# Uses weekly pivot levels (R3/S3) as key institutional levels, daily EMA50 for trend, and volume spike for confirmation.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following institutional flow.
# Target: 15-25 trades/year on 6h to stay within optimal range.

name = "6h_WeeklyPivot_Breakout_Trend_Volume"
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

    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Weekly data for pivot levels (using daily data to calculate weekly pivots)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points: P = (H+L+C)/3, R3 = H + 2*(P-L), S3 = L - 2*(H-P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r3 = weekly_high + 2.0 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2.0 * (weekly_high - weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3, additional_delay_bars=0)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3, additional_delay_bars=0)

    # Volume confirmation: volume > 2.0x 50-period average
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(vol_avg_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above weekly R3 + daily EMA50 uptrend + volume spike
            if (close[i] > weekly_r3_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_50[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S3 + daily EMA50 downtrend + volume spike
            elif (close[i] < weekly_s3_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_50[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly pivot (mean reversion to weekly mean)
            if close[i] < weekly_pivot[-1] if len(weekly_pivot) > 0 else weekly_pivot[0]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly pivot (mean reversion to weekly mean)
            if close[i] > weekly_pivot[-1] if len(weekly_pivot) > 0 else weekly_pivot[0]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals