#!/usr/bin/env python3
# 12h_1D_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: Trade 12h timeframe using daily Camarilla R3/S3 breakouts with 1d trend filter and volume confirmation. 
# Long when price breaks above daily R3 with bullish 1d trend (close > EMA34) and volume spike.
# Short when price breaks below daily S3 with bearish 1d trend (close < EMA34) and volume spike.
# Exit when price crosses back through the opposite level (S3 for longs, R3 for shorts).
# Uses daily trend to avoid counter-trend trades in choppy markets. Low-frequency design (12-37 trades/year) to minimize fee drag.
# Works in bull markets via breakout momentum and in bear markets via short-side breakdowns.

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

    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate daily Camarilla levels (R3 and S3) based on previous day
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Align daily levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Daily trend filter
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price crosses above R3 with bullish daily trend and volume confirmation
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S3 with bearish daily trend and volume confirmation
            elif close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3
            if close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3
            if close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals