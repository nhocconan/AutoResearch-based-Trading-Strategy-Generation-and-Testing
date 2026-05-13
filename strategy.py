#!/usr/bin/env python3
# 4h_Donchian20_Breakout_Volume_ADX
# Hypothesis: Donchian(20) breakouts with volume confirmation and ADX trend filter capture strong momentum moves.
# Works in bull markets by catching breakouts; works in bear markets by filtering weak moves via ADX > 25.
# Uses discrete position sizing (0.25) to limit turnover and fee drag. Designed for ~25-40 trades/year.

name = "4h_Donchian20_Breakout_Volume_ADX"
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

    # Calculate Donchian channels (20-period) on close
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # ADX trend filter (14-period) - requires ADX > 25 for strong trend
    # Calculate +DI, -DI, TR
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.full(n, np.nan)
    for i in range(1, n):
        if i == 1:
            atr[i] = tr[i]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing

    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    for i in range(1, n):
        if atr[i] > 0:
            plus_di[i] = (plus_dm[i] / atr[i]) * 100
            minus_di[i] = (minus_dm[i] / atr[i]) * 100
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100

    adx = np.full(n, np.nan)
    for i in range(14, n):
        valid_dx = dx[14:i+1][~np.isnan(dx[14:i+1])]
        if len(valid_dx) >= 14:
            adx[i] = np.mean(valid_dx[-14:])
        else:
            adx[i] = np.nan

    # ADX > 25 indicates strong trend
    strong_trend = adx > 25

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian high + volume spike + strong trend
            if close[i] > donchian_high[i] and volume_spike[i] and strong_trend[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low + volume spike + strong trend
            elif close[i] < donchian_low[i] and volume_spike[i] and strong_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals