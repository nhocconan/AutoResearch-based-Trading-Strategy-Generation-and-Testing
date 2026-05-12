#!/usr/bin/env python3
# 12h_WeeklyPivot_CamarillaBreakout_TrendVolume
# Hypothesis: Breakout at Camarilla R3/S3 levels from weekly pivot on 12h timeframe with 1d trend filter and volume spike confirmation.
# Uses weekly pivot points (calculated from prior week's OHLC) to derive Camarilla levels (R3, S3) as institutional support/resistance.
# Trend filter: 1d EMA34 ensures alignment with higher timeframe momentum.
# Volume spike: 2x 20-period SMA to confirm institutional participation.
# Designed for low trade frequency (<15/year) to minimize fee drag while capturing significant breakouts in both bull and bear markets.
# Exit on reversal to opposite Camarilla level (R3/S3) or trend violation.

name = "12h_WeeklyPivot_CamarillaBreakout_TrendVolume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)

    # Calculate weekly pivot points and Camarilla levels
    # Pivot = (H + L + C) / 3
    # Camarilla: R3 = Pivot + (H - L) * 1.1 / 2, S3 = Pivot - (H - L) * 1.1 / 2
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    camarilla_r3 = pivot + (weekly_high - weekly_low) * 1.1 / 2
    camarilla_s3 = pivot - (weekly_high - weekly_low) * 1.1 / 2
    
    # Align weekly Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s3)

    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)

    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)

    # Calculate volume spike threshold (2.0x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_daily_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 with uptrend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_daily_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with downtrend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_daily_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 or trend turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema34_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 or trend turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema34_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals