#!/usr/bin/env python3
"""
12h_ThreeFactor_TrendV2
Hypothesis: Combine three robust factors for 12h timeframe - 1) Donchian channel breakout for trend strength, 
2) Volume spike for conviction, and 3) 1d RSI for momentum filtering. This creates a high-conviction 
signal with low trade frequency suitable for 12h timeframe. Works in both bull (breakouts up) and 
bear (breakouts down) markets by requiring volume confirmation and momentum alignment.
"""

name = "12h_ThreeFactor_TrendV2"
timeframe = "12h"
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

    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Donchian channels (20-period) on 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Calculate RSI(14) on 1d data
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)

    # Volume confirmation: volume > 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + RSI not overbought
            if close[i] > donchian_high[i] and volume[i] > vol_avg_20[i] * 1.8 and rsi_1d_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + RSI not oversold
            elif close[i] < donchian_low[i] and volume[i] > vol_avg_20[i] * 1.8 and rsi_1d_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or RSI overbought
            if close[i] < donchian_low[i] or rsi_1d_aligned[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or RSI oversold
            if close[i] > donchian_high[i] or rsi_1d_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals