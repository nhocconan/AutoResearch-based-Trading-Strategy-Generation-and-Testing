#!/usr/bin/env python3
# 4h_Keltner_Channel_Squeeze_Breakout
# Hypothesis: Identifies low volatility squeeze (BB width < KC width) followed by breakout in direction of 1d EMA50 trend.
# Works in bull/bear markets by using trend filter; squeeze reduces false breakouts.
# Entry: BB inside KC + price breaks KC upper/lower + volume > 1.5x average + 1d EMA50 trend alignment.
# Exit: Opposite KC breach or trend reversal. Targets 20-40 trades/year to minimize fee drag.

name = "4h_Keltner_Channel_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower

    # Keltner Channel (20, 2)
    ema_20_kc = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.abs(high - low)).rolling(window=20, min_periods=20).mean().values
    kc_upper = ema_20_kc + 2 * atr
    kc_lower = ema_20_kc - 2 * atr
    kc_width = kc_upper - kc_lower

    # Squeeze condition: BB inside KC
    squeeze = (bb_width < kc_width)

    # Volume confirmation: current volume > 1.5x average of last 6 periods
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]

        if position == 0:
            # LONG: Squeeze release + price breaks above KC upper + volume + uptrend
            if squeeze[i-1] and close[i] > kc_upper[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Squeeze release + price breaks below KC lower + volume + downtrend
            elif squeeze[i-1] and close[i] < kc_lower[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below KC lower OR trend reverses
            if close[i] < kc_lower[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above KC upper OR trend reverses
            if close[i] > kc_upper[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals