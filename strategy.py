#!/usr/bin/env python3
"""
4H_PriceChannel_VolumeRegime_v1
Hypothesis: Combines Donchian(20) breakout with volume confirmation and 
Choppiness Index regime filter to capture trends while avoiding choppy markets.
Uses 4h timeframe with 1d Choppiness Index for regime detection. 
Breakouts only trigger when volume > 1.5x 20-period average and market is trending (CHOP < 38.2).
Targets 50-150 trades over 4 years with low frequency to minimize fee drag.
Works in both bull (breakouts up) and bear (breakouts down) markets.
"""

name = "4H_PriceChannel_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Choppiness Index (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian Channel (20-period) ---
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # --- Choppiness Index (14-period) on 1d ---
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR14
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR14 over 14 periods
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    # Max(high14) - min(low14)
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range14 = max_high14 - min_low14
    
    # Choppiness Index
    chop = 100 * (np.log10(sum_atr14 / (range14 + 1e-10)) / np.log10(14))
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    
    # Align 1d Choppiness Index to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if Choppiness Index is invalid
        if chop_aligned[i] == 0:  # NaN replacement value
            if position != 0:
                # Exit on Donchian reversal
                if position == 1 and close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions: Donchian breakout + volume spike + trending market (CHOP < 38.2)
        long_entry = (close[i] > donchian_high[i]) and vol_spike[i] and (chop_aligned[i] < 38.2)
        short_entry = (close[i] < donchian_low[i]) and vol_spike[i] and (chop_aligned[i] < 38.2)
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on Donchian reversal or market becomes choppy (CHOP > 61.8)
            if position == 1:
                if (close[i] < donchian_low[i]) or (chop_aligned[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > donchian_high[i]) or (chop_aligned[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals