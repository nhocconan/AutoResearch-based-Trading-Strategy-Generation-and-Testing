#!/usr/bin/env python3
# 4h_Donchian_Breakout_Trend_Volume
# Hypothesis: Price breaking out of Donchian Channels with trend confirmation and volume spike captures strong momentum moves in both bull and bear markets.
# Donchian channels adapt to volatility and provide clear breakout levels.
# Entry: Long when high breaks above 20-period Donchian upper + price > EMA50 + volume spike; Short when low breaks below Donchian lower + price < EMA50 + volume spike.
# Exit: Mean reversion to 20-period EMA to avoid overstaying in extended moves.
# Target: 20-30 trades/year on 4h to stay within optimal range while capturing significant moves.

name = "4h_Donchian_Breakout_Trend_Volume"
timeframe = "4h"
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

    # Calculate Donchian Channels (20-period high/low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # EMA20 for exit (mean reversion target)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: volume > 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema20[i]) or np.isnan(ema50[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: High breaks above Donchian upper + EMA50 uptrend + volume spike
            if (high[i] > donch_high[i] and 
                close[i] > ema50[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Low breaks below Donchian lower + EMA50 downtrend + volume spike
            elif (low[i] < donch_low[i] and 
                  close[i] < ema50[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to EMA20 (middle band)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to EMA20 (middle band)
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals