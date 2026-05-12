#!/usr/bin/env python3
# 6h_1D_1W_Camarilla_R4S4_Breakout_Trend_Filter
# Hypothesis: Breakout at weekly Camarilla R4/S4 levels with daily trend filter and volume confirmation.
# Uses weekly structure for direction (more stable in bear/bull) and daily timeframe for entry timing.
# Targets 15-25 trades/year on 6h timeframe to avoid excessive fee drag. Works in both bull and bear markets
# by requiring alignment with higher timeframe trend and volume confirmation to avoid false breakouts.

name = "6h_1D_1W_Camarilla_R4S4_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for Camarilla R4/S4 levels (structure)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Get daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Camarilla R4 and S4 levels from previous weekly OHLC
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values

    camarilla_r4_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 2
    camarilla_s4_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 2

    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/both EMAs for long, below both for short
        bullish_trend = close[i] > ema_1w_aligned[i] and close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i] and close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Break above weekly Camarilla R4 with bullish trend on both timeframes and volume
            if (close[i] > camarilla_r4_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Camarilla S4 with bearish trend on both timeframes and volume
            elif (close[i] < camarilla_s4_aligned[i] and bearish_trend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R4 or trend turns bearish on either timeframe
            if close[i] < camarilla_r4_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or trend turns bullish on either timeframe
            if close[i] > camarilla_s4_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals