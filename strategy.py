#!/usr/bin/env python3
name = "6h_1w_1d_EquiDepth_Channel_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Equi-Depth Channel (Quantile-based)
    # 20-period quantiles on weekly close
    close_1w = df_1w['close'].values
    q20 = pd.Series(close_1w).rolling(window=20, min_periods=20).quantile(0.2).values
    q80 = pd.Series(close_1w).rolling(window=20, min_periods=20).quantile(0.8).values
    
    # Align weekly quantiles to 6h timeframe
    q20_aligned = align_htf_to_ltf(prices, df_1w, q20)
    q80_aligned = align_htf_to_ltf(prices, df_1w, q80)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily Volume Spike (5-period avg)
    vol_ma_5 = pd.Series(df_1d['volume']).rolling(window=5, min_periods=5).mean().values
    vol_ma_5_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 5)  # Wait for weekly quantiles and daily volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(q20_aligned[i]) or np.isnan(q80_aligned[i]) or 
            np.isnan(vol_ma_5_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Q80 with daily volume spike
            vol_condition = df_1d['volume'].iloc[i] > vol_ma_5_aligned[i] * 2.0
            
            if close[i] > q80_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Q20 with daily volume spike
            elif close[i] < q20_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below weekly Q80
            if close[i] < q80_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above weekly Q20
            if close[i] > q20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Equi-Depth Channel Breakout with weekly quantiles and daily volume confirmation
# - Weekly Q20/Q80 act as dynamic support/resistance based on price distribution
# - Breakout above Q80 with 2x daily volume = long opportunity
# - Breakdown below Q20 with 2x daily volume = short opportunity
# - Uses actual weekly quantiles (not fixed levels) to adapt to volatility regimes
# - Volume confirmation ensures institutional participation
# - Works in both bull (buy Q80 breaks) and bear (sell Q20 breaks) via breakout logic
# - Exit when price returns to the opposite quantile
# - Position size 0.25 targets ~50-100 trades over 4 years (~12-25/year)