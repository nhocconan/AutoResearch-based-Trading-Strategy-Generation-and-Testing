#!/usr/bin/env python3
"""
1D_Weekly_RangeBreakout_v2
Hypothesis: Uses weekly price range (high-low) to detect consolidation periods.
When price breaks above the weekly high with volume confirmation, go long.
When price breaks below the weekly low with volume confirmation, go short.
Uses 1-week trend filter (1d timeframe EMA34) to avoid counter-trend trades.
Designed for low trade frequency by requiring both range breakout and volume spike.
Works in both bull and bear markets by following the intermediate-term trend.
"""

name = "1D_Weekly_RangeBreakout_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for range and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly High and Low from 1w data ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly levels to 1d
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- Weekly Trend Filter (EMA34 on 1w close) ---
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above weekly high with volume, above weekly EMA
            if (close[i] > high_1w_aligned[i] and 
                volume_spike and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume, below weekly EMA
            elif (close[i] < low_1w_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of volume
            if position == 1:
                # Exit long: price breaks below weekly low
                if close[i] < low_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above weekly high
                if close[i] > high_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals