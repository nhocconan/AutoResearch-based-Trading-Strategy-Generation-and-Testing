#!/usr/bin/env python3

# 12h_1D_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: Breakout above Camarilla R3 or below S3 on 12h with 1d trend filter and volume confirmation.
# Camarilla levels derived from 1d provide strong support/resistance. Works in both bull and bear markets
# by requiring trend alignment and volume confirmation to avoid false breakouts. Targets 15-30 trades/year.

name = "12h_1D_Camarilla_R3S3_Breakout_Trend_Volume"
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
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate Camarilla levels from previous 1d bar
    # H, L, C from previous 1d bar
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    # Shift by 1 to use previous day's data
    ph = np.roll(d_high, 1)
    pl = np.roll(d_low, 1)
    pc = np.roll(d_close, 1)
    ph[0] = pl[0] = pc[0] = np.nan  # First value invalid

    # Camarilla R3 and S3
    camarilla_r3 = pc + 1.1 * (ph - pl) / 6
    camarilla_s3 = pc - 1.1 * (ph - pl) / 6

    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Calculate 1d EMA for trend filter
    ema_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
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
            # LONG: Break above Camarilla R3 with bullish trend and volume confirmation
            if close[i] > camarilla_r3_aligned[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S3 with bearish trend and volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla R3 or trend turns bearish
            if close[i] < camarilla_r3_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla S3 or trend turns bullish
            if close[i] > camarilla_s3_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals