#!/usr/bin/env python3
# 4h_CombinedBreakout_12hTrend_Volume
# Hypothesis: Price breakout from Donchian(20) or Keltner(20,2) channels on 4h, confirmed by 12h trend direction and volume spikes.
# Combines two breakout systems to increase robustness while maintaining low trade frequency.
# Trend filter: 12h EMA50 (only trade in direction of higher timeframe trend).
# Volume confirmation: current volume > 2.0 x 20-period average.
# Designed to work in both bull and bear markets by following 12h trend direction.
# Target: 20-40 trades/year per symbol to minimize fee drift.

name = "4h_CombinedBreakout_12hTrend_Volume"
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

    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')

    # Calculate Donchian Channels for 4h: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Calculate Keltner Channels for 4h: EMA(20) +/- ATR(10) * 2
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_upper = ema20 + 2 * atr10
    keltner_lower = ema20 - 2 * atr10

    # Trend filter: 12h EMA50
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above either channel in uptrend with volume spike
            if ((close[i] > donchian_high[i] or close[i] > keltner_upper[i]) and
                close[i] > ema50_12h_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below either channel in downtrend with volume spike
            elif ((close[i] < donchian_low[i] or close[i] < keltner_lower[i]) and
                  close[i] < ema50_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below either channel or trend turns down
            if (close[i] < donchian_low[i] or close[i] < keltner_lower[i] or
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above either channel or trend turns up
            if (close[i] > donchian_high[i] or close[i] > keltner_upper[i] or
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals