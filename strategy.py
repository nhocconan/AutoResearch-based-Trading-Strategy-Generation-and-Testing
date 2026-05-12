#!/usr/bin/env python3
# 1d_1W_Camarilla_R4S4_Breakout_TrendFilter_Volume
# Hypothesis: Breakouts at weekly Camarilla R4/S4 levels with daily trend filter and volume confirmation.
# Uses weekly timeframe for structural levels, daily for trend and volume to avoid false breakouts.
# Designed to work in both bull and bear markets by requiring alignment with higher timeframe trend.
# Targets 10-25 trades/year to minimize fee drag.

name = "1d_1W_Camarilla_R4S4_Breakout_TrendFilter_Volume"
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

    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Get daily data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate weekly Camarilla R4 and S4 levels from previous week OHLC
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values

    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2

    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)

    # Volume confirmation: current volume > 1.5x average of last 10 days
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price relative to daily EMA50
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Break above weekly R4 with bullish daily trend and volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S4 with bearish daily trend and volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and bearish_trend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R4 or trend turns bearish
            if close[i] < camarilla_r4_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or trend turns bullish
            if close[i] > camarilla_s4_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals