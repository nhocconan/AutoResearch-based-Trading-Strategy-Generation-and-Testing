#!/usr/bin/env python3
# 4h_TRIX_Trend_Filter_With_Volume_Spike
# Hypothesis: TRIX (Triple Exponential Average) on 12h timeframe filters trend direction (bullish when TRIX > 0, bearish when TRIX < 0),
# while breakout from 4h Donchian channels (20-period) captures momentum entries. Volume spike (volume > 2.0 * 20-period MA) confirms
# institutional participation. The combination reduces whipsaws in sideways markets and captures strong trends.
# Designed for low-frequency, high-quality setups with proper risk management via trend-following exits.

name = "4h_TRIX_Trend_Filter_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for TRIX trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values

    # Calculate TRIX (15,9,9) on 12h close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9) - then % change
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_raw = ema3.pct_change() * 100  # Percentage change
    trix_12h = trix_raw.fillna(0).values  # Fill NaN with 0 for stability
    trix_12h_aligned = align_htf_to_ltf(prices, df_12h, trix_12h)

    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume spike: volume > 2.0 * 20-period average (~3.3 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish TRIX + breakout above Donchian high + volume spike
            if trix_12h_aligned[i] > 0 and close[i] > donchian_high[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish TRIX + breakdown below Donchian low + volume spike
            elif trix_12h_aligned[i] < 0 and close[i] < donchian_low[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns bearish or price breaks below Donchian low
            if trix_12h_aligned[i] < 0 or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns bullish or price breaks above Donchian high
            if trix_12h_aligned[i] > 0 or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals