#!/usr/bin/env python3
# 6h_Keltner_Channel_MeanReversion_1wTrend
# Hypothesis: Price tends to revert to the mean within Keltner Channel bands during ranging markets, but trends strongly when breaking out with weekly trend confirmation. Long when price touches lower band with bullish weekly trend and volume expansion, short when price touches upper band with bearish weekly trend and volume expansion. Uses ATR-based bands for dynamic support/resistance, avoiding whipsaws in strong trends while capturing mean reversion in ranges. Designed for 6h timeframe to balance signal frequency and reliability, targeting 50-150 trades over 4 years.

name = "6h_Keltner_Channel_MeanReversion_1wTrend"
timeframe = "6h"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Keltner Channel (20, 2.0)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_band = ema_20 + (2.0 * atr)
    lower_band = ema_20 - (2.0 * atr)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]

        if position == 0:
            # LONG: Price touches lower Keltner band with weekly uptrend and volume expansion
            if close[i] <= lower_band[i] and weekly_uptrend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper Keltner band with weekly downtrend and volume expansion
            elif close[i] >= upper_band[i] and weekly_downtrend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to EMA(20) or weekly trend turns down
            if close[i] >= ema_20[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to EMA(20) or weekly trend turns up
            if close[i] <= ema_20[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals