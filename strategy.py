#!/usr/bin/env python3

# 4h_1W_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: Buy near Camarilla R3 in uptrend, sell near S3 in downtrend on 4-hour timeframe with weekly trend filter and volume confirmation.
# Camarilla levels provide precise support/resistance. Weekly trend ensures alignment with higher timeframe momentum.
# Volume confirmation filters breakouts. Designed for low frequency (15-30 trades/year) to minimize fee drag.

name = "4h_1W_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "4h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate 4-hour Camarilla levels (based on previous 4h bar)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 4
    S3 = prev_close - rang * 1.1 / 4

    # Volume confirmation: current volume > 1.5x average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or
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
            # LONG: Price crosses above R3 with bullish weekly trend and volume confirmation
            if close[i] > R3[i] and close[i-1] <= R3[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below S3 with bearish weekly trend and volume confirmation
            elif close[i] < S3[i] and close[i-1] >= S3[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 or weekly trend turns bearish
            if close[i] < S3[i] and close[i-1] >= S3[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 or weekly trend turns bullish
            if close[i] > R3[i] and close[i-1] <= R3[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals