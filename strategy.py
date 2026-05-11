#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_Filter
Hypothesis: Uses weekly trend filter (price above/below weekly SMA200) to determine directional bias,
then enters on 6h breakouts of daily Camarilla R4/S4 levels with volume confirmation.
Weekly trend filter avoids counter-trend trades in strong trends, while R4/S4 breakouts capture
momentum moves. Designed for low trade frequency (15-35/year) by requiring multi-timeframe confluence.
Works in bull markets (follow weekly uptrend) and bear markets (follow weekly downtrend).
"""

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily data for Camarilla levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_val_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels (R4 and S4 - stronger breakout levels)
    R4_1d = pivot_1d + (range_val_1d * 1.1 / 2)
    S4_1d = pivot_1d - (range_val_1d * 1.1 / 2)
    
    # Align to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # --- Weekly trend filter (SMA200) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Weekly SMA200
    sma_200_1w = pd.Series(df_1w['close']).rolling(window=200, min_periods=200).mean().values
    sma_200_6h = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # --- Volume spike detection (24-period average on 6h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for weekly SMA200)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(sma_200_6h[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above R4 with volume, above weekly SMA200 (uptrend)
            if (close[i] > R4_6h[i] and 
                volume_spike and 
                close[i] > sma_200_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume, below weekly SMA200 (downtrend)
            elif (close[i] < S4_6h[i] and 
                  volume_spike and 
                  close[i] < sma_200_6h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or trend reversal
            if position == 1:
                # Exit long: price breaks below S4 OR weekly trend turns down
                if (close[i] < S4_6h[i] or 
                    close[i] < sma_200_6h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R4 OR weekly trend turns up
                if (close[i] > R4_6h[i] or 
                    close[i] > sma_200_6h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals