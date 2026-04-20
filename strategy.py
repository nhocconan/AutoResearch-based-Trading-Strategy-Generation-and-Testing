#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour volume confirmation and ATR stop
# - Enter long when price breaks above Donchian(20) high on 4h
# - Enter short when price breaks below Donchian(20) low on 4h
# - Require 12h volume > 1.5x 20-period average for confirmation
# - Exit on opposite Donchian breakout or ATR-based stop
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Donchian calculation (already 4h but using helper for consistency)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels on 4h
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate ATR for stop loss
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_ratio = volume_12h[i // 48] / vol_ma_20_aligned[i] if i >= 20 and not np.isnan(vol_ma_20_aligned[i]) else 0
        vol_confirmed = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume confirmation
            if close_4h[i] > high_20[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            # Short entry: price breaks below Donchian low + volume confirmation
            elif close_4h[i] < low_20[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or ATR stop
            if close_4h[i] < low_20[i] or close_4h[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or ATR stop
            if close_4h[i] > high_20[i] or close_4h[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Volume_ATR_Stop"
timeframe = "4h"
leverage = 1.0