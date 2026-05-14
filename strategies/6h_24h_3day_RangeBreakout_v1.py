#!/usr/bin/env python3
"""
6h_24h_3day_RangeBreakout_v1
Hypothesis: Uses 24-hour price range (high-low) to detect consolidation periods.
When price breaks above the 24h high with volume confirmation, go long.
When price breaks below the 24h low with volume confirmation, go short.
Uses 3-day trend filter (1d timeframe EMA34) to avoid counter-trend trades.
Designed for low trade frequency by requiring both range breakout and volume spike.
Works in both bull and bear markets by following the intermediate-term trend.
"""

name = "6h_24h_3day_RangeBreakout_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for range and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 24h High and Low from 1d data ---
    high_24h = df_1d['high'].values
    low_24h = df_1d['low'].values
    
    # Align 24h levels to 6h
    high_24h_aligned = align_htf_to_ltf(prices, df_1d, high_24h)
    low_24h_aligned = align_htf_to_ltf(prices, df_1d, low_24h)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- 3-day Trend Filter (EMA34 on 1d close) ---
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_24h_aligned[i]) or np.isnan(low_24h_aligned[i]) or
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
            # Long: price breaks above 24h high with volume, above 3-day EMA
            if (close[i] > high_24h_aligned[i] and 
                volume_spike and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24h low with volume, below 3-day EMA
            elif (close[i] < low_24h_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of volume
            if position == 1:
                # Exit long: price breaks below 24h low
                if close[i] < low_24h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above 24h high
                if close[i] > high_24h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals