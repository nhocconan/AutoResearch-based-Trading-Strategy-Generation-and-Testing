#!/usr/bin/env python3
"""
12h_Donchian20_TrendVolume
Hypothesis: Combines 20-period Donchian channel breakouts with 1-day EMA trend filter and volume confirmation.
Long when price breaks above Donchian high with volume > 1.5x average and price above 1-day EMA50.
Short when price breaks below Donchian low with volume > 1.5x average and price below 1-day EMA50.
Designed for low trade frequency (target: 20-50 trades/year) by requiring confluence of three conditions.
Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with trend filter.
"""

name = "12h_Donchian20_TrendVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian Channel (20-period) ---
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- 1-day EMA50 Trend Filter ---
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
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
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, above 1-day EMA50
            if (close[i] > high_max[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, below 1-day EMA50
            elif (close[i] < low_min[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout
            if position == 1:
                # Exit long: price breaks below Donchian low
                if close[i] < low_min[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above Donchian high
                if close[i] > high_max[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals