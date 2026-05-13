#!/usr/bin/env python3
# 1h_Engulfing_4hTrend_1dVolFilter
# Hypothesis: Bullish/bearish engulfing candles on 1h capture momentum shifts, filtered by 4h trend (EMA50) and 1d volume surge to avoid false signals.
# Works in bull markets by taking longs in uptrends and in bear markets by taking shorts in downtrends.
# Volume filter ensures trades occur during periods of heightened interest, reducing whipsaw.
# Target: 20-30 trades/year per symbol.

name = "1h_Engulfing_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values

    # Calculate 4h EMA50 for trend
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)

    # Calculate 1d average volume (20-period) for surge filter
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-20:i])
    vol_surge_1d = volume_1d > (1.5 * vol_avg_1d)
    vol_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_surge_1d.astype(float))

    # Detect engulfing candles on 1h
    bullish_engulfing = (close > open_price) & (open_price > np.roll(close, 1)) & (close > np.roll(open_price, 1))
    bearish_engulfing = (close < open_price) & (open_price < np.roll(close, 1)) & (close < np.roll(open_price, 1))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_surge_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: bullish engulfing + 4h uptrend + 1d volume surge
            if bullish_engulfing[i] and close[i] > ema_4h_aligned[i] and vol_surge_1d_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # SHORT: bearish engulfing + 4h downtrend + 1d volume surge
            elif bearish_engulfing[i] and close[i] < ema_4h_aligned[i] and vol_surge_1d_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish engulfing or trend reversal
            if bearish_engulfing[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: bullish engulfing or trend reversal
            if bullish_engulfing[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals