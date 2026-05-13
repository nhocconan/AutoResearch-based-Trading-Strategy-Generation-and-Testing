#!/usr/bin/env python3
# 6h_Keltner_Breakout_1dTrend_VolumeSpike
# Hypothesis: Keltner channel breakouts capture momentum with less whipsaw than Bollinger Bands.
# Enter long when price breaks above upper Keltner band with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below lower Keltner band with volume spike and 1d EMA50 downtrend.
# Exit when price re-enters the Keltner channel.
# Uses 6h timeframe with 1d trend filter to balance trade frequency and win rate.
# Designed to work in both bull (buy in uptrend) and bear (sell in downtrend).
# Target: 15-30 trades/year per symbol.

name = "6h_Keltner_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
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
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Keltner Channel (20, 2) on 6h data
    # Typical Price
    tp = (high + low + close) / 3
    # ATR (10)
    tr0 = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr[0] = tr0[0]
    atr = np.zeros(n)
    for i in range(10, n):
        atr[i] = np.mean(tr[i-10:i])
    # EMA of Typical Price (20)
    ema_tp = pd.Series(tp).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Keltner Bands
    keltner_up = ema_tp + 2 * atr
    keltner_low = ema_tp - 2 * atr

    # Breakout signals
    buy_signal = close > keltner_up
    sell_signal = close < keltner_low
    reentry_signal = (close >= keltner_low) & (close <= keltner_up)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(keltner_up[i]) or np.isnan(keltner_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Keltner band with volume spike and 1d EMA uptrend
            if buy_signal[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner band with volume spike and 1d EMA downtrend
            elif sell_signal[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Keltner channel
            if reentry_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Keltner channel
            if reentry_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals