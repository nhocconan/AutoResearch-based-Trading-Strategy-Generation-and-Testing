#!/usr/bin/env python3
# 1d_Keltner_Channel_WeeklyTrend_Filter
# Hypothesis: On daily timeframe, use Keltner Channel (ATR-based) breakout for entry in the direction of weekly EMA50 trend.
# Exit on opposite Keltner Channel touch. Volume confirmation required (>1.5x 20-day average volume).
# Designed for low frequency (~15-25 trades/year) with clear trend following logic.
# Works in bull markets (breakouts with trend) and bear markets (follows weekly trend, avoids counter-trend whipsaws).

name = "1d_Keltner_Channel_WeeklyTrend_Filter"
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

    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)

    close_weekly = df_weekly['close'].values
    # Calculate EMA50 on weekly close
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)

    # Calculate daily ATR(10) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Calculate Keltner Channel: EMA20 ± 2*ATR
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr

    # Volume confirmation: 1.5x 20-day average volume
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above KC upper in uptrend (weekly close > EMA50) with volume
            if (close[i] > kc_upper[i] and 
                close[i-1] <= kc_upper[i-1] and  # Confirm breakout
                close[i] > ema50_weekly_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below KC lower in downtrend (weekly close < EMA50) with volume
            elif (close[i] < kc_lower[i] and 
                  close[i-1] >= kc_lower[i-1] and  # Confirm breakdown
                  close[i] < ema50_weekly_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or crosses KC lower (mean reversion)
            if close[i] <= kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or crosses KC upper (mean reversion)
            if close[i] >= kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals