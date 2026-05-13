#!/usr/bin/env python3
# 1d_Engulfing_1wTrend_VolumeConfirm
# Hypothesis: Bullish/bearish engulfing candles on daily timeframe, filtered by weekly trend and volume confirmation, capture high-probability reversals in both bull and bear markets. Weekly trend ensures alignment with higher-timeframe momentum, reducing false signals. Volume confirms conviction. Target: 10-25 trades/year.

name = "1d_Engulfing_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA20 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate 20-period volume average for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if data is not ready
        if np.isnan(vol_ma[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Detect bullish engulfing: current green candle fully engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                         (close[i] >= open_price[i-1]) and (open_price[i] <= close[i-1])
        # Detect bearish engulfing: current red candle fully engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                         (open_price[i] >= close[i-1]) and (close[i] <= open_price[i-1])

        if position == 0:
            # LONG: Bullish engulfing with volume confirmation and weekly uptrend
            if bullish_engulf and volume_confirm[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish engulfing with volume confirmation and weekly downtrend
            elif bearish_engulf and volume_confirm[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish engulfing or weekly trend turns down
            if bearish_engulf or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish engulfing or weekly trend turns up
            if bullish_engulf or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals