#!/usr/bin/env python3
"""
4H_Donchian20_Breakout_Volume_Regime_v1
Hypothesis: Use 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
Long when price breaks above Donchian(20) high with volume > 1.5x average and chop > 61.8 (range).
Short when price breaks below Donchian(20) low with volume > 1.5x average and chop > 61.8.
This targets mean-reversion in ranging markets while using volume to confirm breakout strength,
working in both bull and bear markets by focusing on range-bound conditions.
"""
name = "4H_Donchian20_Breakout_Volume_Regime_v1"
timeframe = "4h"
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for Chop index (range detection)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Chop index (14-period)
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    atr_1d = []
    tr_list = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d.iloc[i] - low_1d.iloc[i]
        else:
            tr = max(
                high_1d.iloc[i] - low_1d.iloc[i],
                abs(high_1d.iloc[i] - close_1d.iloc[i-1]),
                abs(low_1d.iloc[i] - close_1d.iloc[i-1])
            )
        tr_list.append(tr)
        if len(tr_list) < 14:
            atr_1d.append(np.nan)
        else:
            atr_val = np.mean(tr_list[-14:])
            atr_1d.append(atr_val)
    
    atr_1d = np.array(atr_1d)
    sum_high_low_1d = (high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low_1d / (atr_1d * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 14)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 16 bars between trades (4 days on 4h TF) to reduce frequency
            if bars_since_exit < 16:
                continue
                
            # Long: price breaks above Donchian high with volume and chop > 61.8 (range)
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below Donchian low with volume and chop > 61.8 (range)
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level
            if position == 1 and close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals