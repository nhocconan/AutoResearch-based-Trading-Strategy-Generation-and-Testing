#!/usr/bin/env python3
"""
1d_Keltner_Breakout_WeeklyTrend_Volume
Hypothesis: Breakouts above/below weekly Keltner upper/lower bands with volume confirmation and weekly trend filter.
Operates on 1d timeframe to target 7-25 trades per year (30-100 total over 4 years).
Uses weekly timeframe for trend direction (EMA50) and volatility (ATR10 for Keltner channels).
Volume confirmation ensures breakout strength. Designed to work in both bull and bear regimes
by following the weekly trend direction, avoiding counter-trend trades.
"""

name = "1d_Keltner_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for Keltner bands and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly ATR(10) for Keltner channels
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(low_1w[1:], high_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Weekly Keltner channels: EMA20 ± 2*ATR10
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema20_1w + 2 * atr10
    keltner_lower = ema20_1w - 2 * atr10

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 1d bar
        keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)[i]
        keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)[i]
        ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(keltner_upper_aligned) or np.isnan(keltner_lower_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above weekly Keltner upper + volume spike (1.5x) + weekly uptrend
            if (close[i] > keltner_upper_aligned and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below weekly Keltner lower + volume spike (1.5x) + weekly downtrend
            elif (close[i] < keltner_lower_aligned and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly Keltner lower (reversal signal)
            if close[i] < keltner_lower_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly Keltner upper (reversal signal)
            if close[i] > keltner_upper_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals