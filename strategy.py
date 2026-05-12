#!/usr/bin/env python3
"""
6h_HeikinAshi_KeltnerTrend_Follow
Hypothesis: In both bull and bear markets, strong trends persist with momentum. Uses Heikin Ashi candles on 6h to filter noise, combined with Keltner Channel breakout and 1d EMA trend alignment. Enters long when HA close > upper Keltner + bullish candle + 1d EMA up; short when HA close < lower Keltner + bearish candle + 1d EMA down. Exits on opposite signal or trend reversal. Designed for low frequency (target 20-40 trades/year) to minimize fee drag.
"""

name = "6h_HeikinAshi_KeltnerTrend_Follow"
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
    open_ = prices['open'].values
    volume = prices['volume'].values

    # --- Heikin Ashi Calculation ---
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.zeros_like(open_)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(np.maximum(high, low), np.maximum(ha_open, ha_close))
    ha_low = np.minimum(np.minimum(high, low), np.minimum(ha_open, ha_close))

    # --- Keltner Channel (20, 2) ---
    # Typical Price
    tp = (high + low + close) / 3
    # EMA of TP for middle line
    ema_tp = pd.Series(tp).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR for bands
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_tp + 2 * atr
    keltner_lower = ema_tp - 2 * atr

    # --- 1d EMA34 for trend filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        # Skip if any required value is NaN
        if (np.isnan(ha_open[i]) or np.isnan(ha_close[i]) or 
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        ha_bullish = ha_close[i] > ha_open[i]
        ha_bearish = ha_close[i] < ha_open[i]

        if position == 0:
            # LONG: HA bullish, close above upper Keltner, 1d EMA up
            if ha_bullish and ha_close[i] > keltner_upper[i] and ha_close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: HA bearish, close below lower Keltner, 1d EMA down
            elif ha_bearish and ha_close[i] < keltner_lower[i] and ha_close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: HA bearish OR close below lower Keltner OR trend turns down
            if ha_bearish or ha_close[i] < keltner_lower[i] or ha_close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: HA bullish OR close above upper Keltner OR trend turns up
            if ha_bullish or ha_close[i] > keltner_upper[i] or ha_close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals