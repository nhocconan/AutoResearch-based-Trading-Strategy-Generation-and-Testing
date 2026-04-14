#!/usr/bin/env python3
"""
12h_1dVolatilityRegime_RangeBreakout
Hypothesis: In low volatility regimes (BB width < 3%), price tends to break out of prior 24h range.
Use 1d Bollinger Band width for regime detection, breakout of prior day's high/low for entry.
Works in both bull/bear markets by capturing volatility expansion after contraction.
Target: 15-25 trades/year to minimize fee drag.
"""

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Bollinger Bands (20, 2) for volatility regime
    sma_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Calculate prior day's high/low for breakout levels
    prior_high_1d = df_1d['high'].shift(1).values  # Prior day's high
    prior_low_1d = df_1d['low'].shift(1).values    # Prior day's low
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high_1d)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(prior_high_aligned[i]) or
            np.isnan(prior_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility regime filter: BB width < 3% of price indicates low volatility/squeeze
        bb_width = (upper_bb_aligned[i] - lower_bb_aligned[i]) / price if price > 0 else 0
        low_vol_regime = bb_width < 0.03
        
        if position == 0:
            # Long breakout: price breaks above prior day's high in low vol regime
            if low_vol_regime and price > prior_high_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short breakout: price breaks below prior day's low in low vol regime
            elif low_vol_regime and price < prior_low_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to prior day's low (mean reversion) or volatility expands
            if price < prior_low_aligned[i] or bb_width > 0.08:  # Exit if volatility expands
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to prior day's high or volatility expands
            if price > prior_high_aligned[i] or bb_width > 0.08:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dVolatilityRegime_RangeBreakout"
timeframe = "12h"
leverage = 1.0