#!/usr/bin/env python3
"""
6h_1d_PolarizedFractal_Efficiency_Trend
Hypothesis: Use Polarized Fractal Efficiency (PFE) on 1d timeframe to identify trending vs ranging markets, then trade breakouts of 6h Donchian channels only when PFE indicates trending. PFE values near +100 indicate strong uptrend, near -100 indicate strong downtrend, and near 0 indicate ranging. This avoids whipsaws in chop while capturing trends. Works in bull markets by buying dips in uptrends (PFE>0) and in bear markets by selling rallies in downtrends (PFE<0). Targets 15-25 trades/year by requiring PFE > |30| for trend confirmation and Donchian breakout in trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for PFE calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Polarized Fractal Efficiency (PFE) over 10 periods
    # PFE = 100 * (close - lowest low in period) / (highest high - lowest low) * 2 - 100
    # Actually: PFE = 100 * sqrt((price change)^2 + (period-1)^2) / sum of absolute daily changes * direction
    # Simplified: PFE = 100 * (net change / sum of absolute changes) where net change = close[t] - close[t-10]
    pfe = np.full(len(close_1d), np.nan)
    lookback = 10
    
    for i in range(lookback, len(close_1d)):
        net_change = close_1d[i] - close_1d[i - lookback]
        abs_sum = 0
        for j in range(1, lookback + 1):
            abs_sum += abs(close_1d[i - j + 1] - close_1d[i - j])
        if abs_sum != 0:
            pfe[i] = 100 * net_change / abs_sum
        else:
            pfe[i] = 0
    
    # Align PFE to 6h timeframe (wait for 1d bar close)
    pfe_aligned = align_htf_to_ltf(prices, df_1d, pfe)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need PFE and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pfe_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: PFE indicates uptrend (>30) and price breaks above Donchian high
            if pfe_aligned[i] > 30 and close[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: PFE indicates downtrend (<-30) and price breaks below Donchian low
            elif pfe_aligned[i] < -30 and close[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: PFE turns negative (trend change) or price returns below Donchian low
            if pfe_aligned[i] < 0 or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: PFE turns positive (trend change) or price returns above Donchian high
            if pfe_aligned[i] > 0 or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_PolarizedFractal_Efficiency_Trend"
timeframe = "6h"
leverage = 1.0