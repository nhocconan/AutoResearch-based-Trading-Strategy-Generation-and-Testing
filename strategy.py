#!/usr/bin/env python3
"""
6h_Weekly_Trend_Follow_v1
Hypothesis: Trend following strategy using weekly trend filter (EMA34) on 1w timeframe
combined with 6h price action for entries. In both bull and bear markets, we trade
in the direction of the weekly trend, capturing major moves while avoiding counter-trend
whipsaws. Uses volume confirmation to filter false breakouts.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Weekly_Trend_Follow_v1"
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
    volume = prices['volume'].values
    
    # === 1W Data for Weekly Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 1D Data for Volume Average ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Daily 20-period average volume for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average volume
        vol_condition = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price above weekly EMA34 AND volume confirmation
            if close[i] > ema34_1w_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA34 AND volume confirmation
            elif close[i] < ema34_1w_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly EMA34
            if close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above weekly EMA34
            if close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals