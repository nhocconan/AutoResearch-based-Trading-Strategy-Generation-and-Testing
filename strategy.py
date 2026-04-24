#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1-week Funding Rate Mean Reversion.
- Williams %R(14) identifies overbought/oversold conditions on 6h chart.
- Weekly funding rate extreme (z-score) provides contrarian bias for BTC/ETH mean reversion.
- Only take longs when funding is extremely negative (bullish bias) and %R oversold.
- Only take shorts when funding is extremely positive (bearish bias) and %R overbought.
- Volume confirmation filters low-quality signals.
- Works in bull/bear markets via funding rate mean reversion edge.
- Targets 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w funding rate data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly funding rate z-score (30-week lookback) for mean reversion
    funding_1w = df_1w['close'].values  # funding rate stored in close column for 1w data
    funding_mean = pd.Series(funding_1w).rolling(window=30, min_periods=15).mean().values
    funding_std = pd.Series(funding_1w).rolling(window=30, min_periods=15).std().values
    funding_z = (funding_1w - funding_mean) / (funding_std + 1e-10)
    funding_z_aligned = align_htf_to_ltf(prices, df_1w, funding_z, additional_delay_bars=1)
    
    # Williams %R (14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 30) + 5
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(funding_z_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and volume_confirm:
            # Long: extremely negative funding (bullish bias) + Williams %R oversold
            if funding_z_aligned[i] < -2.0 and williams_r[i] < -80:
                signals[i] = 0.25
                position = 1
            # Short: extremely positive funding (bearish bias) + Williams %R overbought
            elif funding_z_aligned[i] > 2.0 and williams_r[i] > -20:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: funding normalizes OR %R reaches overbought
            if funding_z_aligned[i] > -0.5 or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: funding normalizes OR %R reaches oversold
            if funding_z_aligned[i] < 0.5 or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_FundingZ_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0