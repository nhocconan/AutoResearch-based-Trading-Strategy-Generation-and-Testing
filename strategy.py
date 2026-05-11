#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend_v1
# Hypothesis: Donchian(20) breakout on 4h timeframe with volume confirmation and 1d trend filter.
# Long when price breaks above 20-period high with volume spike and above 1d EMA50.
# Short when price breaks below 20-period low with volume spike and below 1d EMA50.
# Exit on opposite Donchian breakout.
# Designed for low trade frequency (<50/year) with strong edge in both bull and bear markets.
# Uses 1d EMA50 trend filter to avoid counter-trend trades, improving performance in 2022 bear and 2025 ranging markets.

name = "4h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian Channel (20-period) ---
    # Calculate rolling high/low on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- 1d Trend Filter (EMA50) ---
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (ensure Donchian and EMA are valid)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above 20-period high with volume, above 1d EMA50
            if (close[i] > high_20[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with volume, below 1d EMA50
            elif (close[i] < low_20[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout
            if position == 1:
                # Exit long: price breaks below 20-period low
                if close[i] < low_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above 20-period high
                if close[i] > high_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals