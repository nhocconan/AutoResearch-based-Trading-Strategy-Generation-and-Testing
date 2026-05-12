#!/usr/bin/env python3
# 1d_1W_Keltner_Breakout_Volume_Trend
# Hypothesis: Daily breakouts above weekly Keltner upper band or below lower band with volume confirmation and weekly trend filter.
# Uses weekly timeframe for trend and Keltner bands to reduce noise and avoid overtrading. Designed for 10-30 trades/year.
# Weekly trend filter avoids whipsaws in range markets; volume confirms institutional interest.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following the weekly trend.

name = "1d_1W_Keltner_Breakout_Volume_Trend"
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

    # Get weekly data for trend filter and Keltner bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate weekly Keltner bands: EMA(34) ± 2*ATR(10)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w_arr, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = high_1w[0] - close_1w_arr[0]
    tr3[0] = low_1w[0] - close_1w_arr[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_upper = ema_1w + 2 * atr_1w
    keltner_lower = ema_1w - 2 * atr_1w

    # Align weekly levels to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Price closes above Keltner upper band with bullish weekly trend and volume confirmation
            if close[i] > keltner_upper_aligned[i] and close[i-1] <= keltner_upper_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below Keltner lower band with bearish weekly trend and volume confirmation
            elif close[i] < keltner_lower_aligned[i] and close[i-1] >= keltner_lower_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA or weekly trend turns bearish
            if close[i] < ema_1w_aligned[i] and close[i-1] >= ema_1w_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA or weekly trend turns bullish
            if close[i] > ema_1w_aligned[i] and close[i-1] <= ema_1w_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals