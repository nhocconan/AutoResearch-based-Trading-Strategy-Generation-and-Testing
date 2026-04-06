#!/usr/bin/env python3
"""
12h Camarilla Pivot with Volume Spike and Chop Filter
Hypothesis: Price reverses at Camarilla pivot levels (H3/L3) from prior day.
Volume spike confirms institutional interest. Chop filter avoids whipsaws in ranging markets.
Works in bull (buys at L3 in uptrend) and bear (sells at H3 in downtrend).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "12h_camarilla_pivot_volume_chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    R4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    R2 = close_1d + ((high_1d - low_1d) * 1.1 / 6)
    R1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    S1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    S2 = close_1d - ((high_1d - low_1d) * 1.1 / 6)
    S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    S4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)  # Short entry level
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)  # Long entry level
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Chop filter using EMA crossover (avoid whipsaws)
    ema_fast = pd.Series(close).ewm(span=9, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False).mean().values
    chop_filter = np.abs(ema_fast - ema_slow) > (0.001 * close)  # Trending condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if pivot levels not available
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches S4 (strong support) or stoploss
            if (low[i] <= S3_12h[i] * 0.995 or  # Near S3 level
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R4 (strong resistance) or stoploss
            if (high[i] >= R3_12h[i] * 1.005 or  # Near R3 level
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at Camarilla levels with volume and trend
            long_setup = (low[i] <= S3_12h[i] * 1.005 and  # Touch S3 level
                         vol_spike[i] and chop_filter[i])
            short_setup = (high[i] >= R3_12h[i] * 0.995 and  # Touch R3 level
                          vol_spike[i] and chop_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals