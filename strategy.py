#!/usr/bin/env python3
# 4h_1D_Camarilla_R1S1_Breakout_Trend_Volume
# Hypothesis: Buy when price breaks above daily Camarilla R1 with 4h bullish trend (EMA50) and volume confirmation; sell when breaks below daily Camarilla S1 with 4h bearish trend and volume confirmation. Daily pivot levels provide strong support/resistance; breakouts indicate momentum. Trend filter avoids counter-trend trades. Volume filter reduces false breakouts. Designed for low frequency (20-50 trades/year) to minimize fee drift.

name = "4h_1D_Camarilla_R1S1_Breakout_Trend_Volume"
timeframe = "4h"
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

    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Get daily data for Camarilla levels (R1 and S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily Camarilla levels (R1 and S1) based on previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12

    # Align daily levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # 4h trend filter
        bullish_trend = close[i] > ema_4h_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i]

        if position == 0:
            # LONG: Price crosses above R1 with bullish 4h trend and volume confirmation
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S1 with bearish 4h trend and volume confirmation
            elif close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or 4h trend turns bearish
            if close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or 4h trend turns bullish
            if close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals