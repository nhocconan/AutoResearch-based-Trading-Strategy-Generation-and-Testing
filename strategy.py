#!/usr/bin/env python3

# 12h_1D_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Breakout above/below Camarilla R1/S1 levels on 12h with 1d EMA trend filter and volume confirmation.
# Camarilla levels provide precise support/resistance based on prior day's range, effective in both trending and ranging markets.
# Trend filter ensures alignment with higher timeframe momentum, while volume confirmation avoids false breakouts.
# Targets 15-30 trades/year on 12h timeframe to minimize fee drag.

name = "12h_1D_Camarilla_R1_S1_Breakout_Trend_Filter"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day's range
    # R1 = C + (H-L) * 1.12, S1 = C - (H-L) * 1.12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.12

    # Align Camarilla levels to 12h timeframe (available after previous day closes)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price above Camarilla R1 with bullish trend and volume confirmation
            if close[i] > camarilla_r1_aligned[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Camarilla S1 with bearish trend and volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 or trend turns bearish
            if close[i] < camarilla_s1_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 or trend turns bullish
            if close[i] > camarilla_r1_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals