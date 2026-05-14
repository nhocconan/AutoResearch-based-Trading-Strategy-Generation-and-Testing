#!/usr/bin/env python3

# 4h_1D_Keltner_Breakout_Volume_Momentum
# Hypothesis: Breakout above/below 20-period Keltner Channel on 4h with 1d trend filter and volume confirmation.
# Keltner Channel adapts to volatility, providing dynamic support/resistance. Works in both bull and bear markets
# by requiring trend alignment and volume confirmation to avoid false breakouts. Targets 20-40 trades/year.

name = "4h_1D_Keltner_Breakout_Volume_Momentum"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h Keltner Channel (20-period ATR, 2.0 multiplier)
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # No previous close for first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    ema_middle = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_middle + (2.0 * atr)
    keltner_lower = ema_middle - (2.0 * atr)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i]) or np.isnan(volume_ok[i])):
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
            # LONG: Break above Keltner upper with bullish trend and volume confirmation
            if close[i] > keltner_upper[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Keltner lower with bearish trend and volume confirmation
            elif close[i] < keltner_lower[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below middle line or trend turns bearish
            if close[i] < ema_middle[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above middle line or trend turns bullish
            if close[i] > ema_middle[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals