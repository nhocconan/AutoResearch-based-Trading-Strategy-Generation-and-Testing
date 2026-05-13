#!/usr/bin/env python3
# 6h_HeikinAshi_Trend_Momentum_With_Volume
# Hypothesis: Heikin Ashi candles on 6h filter noise and reveal true trend, while momentum
# (RSI) and volume confirm strength. Works in bull/bear by capturing sustained moves with
# strict entry conditions. Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "6h_HeikinAshi_Trend_Momentum_With_Volume"
timeframe = "6h"
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

    # Heikin Ashi calculation
    ha_close = (open_price + high + low + close) / 4.0
    ha_open = np.zeros(n)
    ha_open[0] = (open_price[0] + close[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])

    # RSI (14) on HA close
    delta = np.diff(ha_close, prepend=ha_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = pd.Series(volume).rolling(20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    # Daily trend filter from 1D close EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(ha_open[i]) or np.isnan(ha_close[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: HA bullish (close > open), RSI > 55, volume spike, price above daily EMA
            if (ha_close[i] > ha_open[i] and 
                rsi[i] > 55 and 
                volume_spike[i] and 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: HA bearish (close < open), RSI < 45, volume spike, price below daily EMA
            elif (ha_close[i] < ha_open[i] and 
                  rsi[i] < 45 and 
                  volume_spike[i] and 
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: HA turns bearish OR RSI < 50
            if ha_close[i] < ha_open[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: HA turns bullish OR RSI > 50
            if ha_close[i] > ha_open[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals