#!/usr/bin/env python3
# 4h_Donchian20_Plus_Volume_Spike_and_KAMA_Direction
# Hypothesis: Combine Donchian(20) breakout with volume confirmation and KAMA direction filter.
# Donchian breakouts capture strong momentum moves; volume spikes confirm institutional participation.
# KAMA adapts to market noise, filtering out false breakouts in choppy conditions.
# Works in both bull and bear markets via adaptive trend filter and volatility-based position sizing.
# Target: 20-50 trades/year per symbol, suitable for 4h timeframe.

name = "4h_Donchian20_Plus_Volume_Spike_and_KAMA_Direction"
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

    # Get daily data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'])).rolling(window=er_len, min_periods=1).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2

    # Calculate KAMA
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'][0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'][i] - kama[i-1])

    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)

    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 2.0x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price relative to KAMA
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]

        if position == 0:
            # LONG: Break above Donchian high + above KAMA + volume spike
            if close[i] > donchian_high[i] and price_above_kama and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + below KAMA + volume spike
            elif close[i] < donchian_low[i] and price_below_kama and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR drops below KAMA
            if close[i] < donchian_low[i] or not price_above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR rises above KAMA
            if close[i] > donchian_high[i] or not price_below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals