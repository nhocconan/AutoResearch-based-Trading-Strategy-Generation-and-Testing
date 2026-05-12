#!/usr/bin/env python3
# 12h_1D_Keltner_Breakout_Trend_Volume
# Hypothesis: Breakouts above/below Keltner Channel (2*ATR) on 12h with 1d trend filter and volume confirmation.
# Works in bull/bear markets: In uptrends, buy upper band breakouts; in downtrends, sell lower band breakouts.
# Volume ensures breakout validity, reducing false signals. Designed for 12h to limit trade frequency.

name = "12h_1D_Keltner_Breakout_Trend_Volume"
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

    # Get 12h data for Keltner Channel and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # 12h ATR(20) for Keltner Channel
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_20_12h = tr.rolling(window=20, min_periods=20).mean().values

    # 12h EMA20 for Keltner middle line
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner upper and lower bands (2*ATR)
    keltner_upper = ema_20_12h + 2 * atr_20_12h
    keltner_lower = ema_20_12h - 2 * atr_20_12h

    # Align Keltner bands to 12h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_12h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_12h, keltner_lower)

    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Get 1d data for trend filter (using EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters
        uptrend_12h = close[i] > ema_50_12h_aligned[i]
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_12h = close[i] < ema_50_12h_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Break above Keltner upper band in uptrend (both timeframes) with volume
            if (close[i] > keltner_upper_aligned[i] and uptrend_12h and uptrend_1d and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Keltner lower band in downtrend (both timeframes) with volume
            elif (close[i] < keltner_lower_aligned[i] and downtrend_12h and downtrend_1d and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Keltner channel (below upper band) or trend reversal
            if close[i] < keltner_upper_aligned[i] or not (uptrend_12h and uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Keltner channel (above lower band) or trend reversal
            if close[i] > keltner_lower_aligned[i] or not (downtrend_12h and downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals