# 6h_WeeklyPivot_DailyTrend_VolumeConfirm
# Hypothesis: On 6h timeframe, trade breakouts of weekly (Monday) Camarilla pivot levels in the direction of the daily trend (EMA34) with volume confirmation (>1.5x average). This combines weekly structure with daily trend alignment to capture sustained moves while avoiding counter-trend noise. Designed for low frequency (target: 20-50 trades/year) to minimize fee drag in both bull and bear markets.

name = "6h_WeeklyPivot_DailyTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Get weekly data for pivot calculation (using Monday's open as week start)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly Camarilla levels from previous week
    # Camarilla: R1 = close + (high - low) * 1.12/12, S1 = close - (high - low) * 1.12/12
    range_1w = high_1w - low_1w
    camarilla_r1_w = close_1w + range_1w * 1.12 / 12
    camarilla_s1_w = close_1w - range_1w * 1.12 / 12
    camarilla_r2_w = close_1w + range_1w * 1.12 / 6
    camarilla_s2_w = close_1w - range_1w * 1.12 / 6

    # Use previous week's levels (shift by 1)
    camarilla_r1_w_prev = np.roll(camarilla_r1_w, 1)
    camarilla_s1_w_prev = np.roll(camarilla_s1_w, 1)
    camarilla_r2_w_prev = np.roll(camarilla_r2_w, 1)
    camarilla_s2_w_prev = np.roll(camarilla_s2_w, 1)
    camarilla_r1_w_prev[0] = np.nan
    camarilla_s1_w_prev[0] = np.nan
    camarilla_r2_w_prev[0] = np.nan
    camarilla_s2_w_prev[0] = np.nan

    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r1_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_w_prev)
    camarilla_s1_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_w_prev)
    camarilla_r2_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r2_w_prev)
    camarilla_s2_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s2_w_prev)

    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Daily volume average (20-period for ~1 month context)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_w_aligned[i]) or np.isnan(camarilla_s1_w_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly R1 + daily uptrend + volume confirmation
            if (close[i] > camarilla_r1_w_aligned[i] and
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 + daily downtrend + volume confirmation
            elif (close[i] < camarilla_s1_w_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly S1 OR trend turns down
            if close[i] < camarilla_s1_w_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly R1 OR trend turns up
            if close[i] > camarilla_r1_w_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals