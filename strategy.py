#!/usr/bin/env python3
"""
6h_Pivot_Reversal_With_Volume_Spike_v2
Hypothesis: Reversal at daily pivot points (PP) with volume spike and rejection of R1/S1 levels.
In ranging markets, price tends to revert to the pivot after testing R1/S1. In trending markets,
breakouts beyond R1/S1 with volume continuation are filtered out by requiring price to stay
within the pivot-R1/S1 band. Uses 6h timeframe to reduce noise and overtrading.
Target: 20-50 trades/year, size 0.25.
"""

name = "6h_Pivot_Reversal_With_Volume_Spike_v2"
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
    
    # Get 1d data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily pivot points: PP = (H + L + C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pp - df_1d['low']
    s1 = 2 * pp - df_1d['high']
    
    # Align to 6h - use previous day's levels (available at 6h open)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Volume confirmation: current volume > 1.5x 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        if position == 0:
            # LONG: Price rejects S1 (bounces off support) with volume spike, below pivot
            if (low[i] <= s1_aligned[i] and  # tested S1
                close[i] > s1_aligned[i] and  # closed above S1 (rejection)
                close[i] < pp_aligned[i] and  # still below pivot (range-bound)
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects R1 (fails at resistance) with volume spike, above pivot
            elif (high[i] >= r1_aligned[i] and  # tested R1
                  close[i] < r1_aligned[i] and  # closed below R1 (rejection)
                  close[i] > pp_aligned[i] and  # still above pivot (range-bound)
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot (mean reversion target) OR stops rejecting S1
            if (close[i] >= pp_aligned[i] or  # reached pivot target
                low[i] > s1_aligned[i]):      # no longer testing S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot (mean reversion target) OR stops rejecting R1
            if (close[i] <= pp_aligned[i] or  # reached pivot target
                high[i] < r1_aligned[i]):     # no longer testing R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals