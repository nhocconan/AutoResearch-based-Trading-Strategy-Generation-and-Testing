#!/usr/bin/env python3
# 6H_1D_DONCHIAN_BREAKOUT_1D_VOLUME_FILTER
# Hypothesis: Breakouts at 6-hour Donchian(20) with 1-day volume confirmation.
# Uses 1-day volume spike (current volume > 1.5x 20-period average) to filter breakouts.
# Works in bull/bear: buy breakouts above upper band, sell breakdowns below lower band.
# Volume confirmation reduces false breakouts. Targets 12-37 trades/year on 6h.

name = "6H_1D_DONCHIAN_BREAKOUT_1D_VOLUME_FILTER"
timeframe = "6h"
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

    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d volume spike filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (1.5 * vol_ma)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)

    # Calculate Donchian channels on 6h data (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper Donchian with volume spike
            if close[i] > high_20[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower Donchian with volume spike
            elif close[i] < low_20[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below lower Donchian (mean reversion)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above upper Donchian (mean reversion)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals