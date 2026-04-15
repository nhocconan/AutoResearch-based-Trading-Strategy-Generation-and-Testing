#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d volatility regime filter
# Uses Donchian channel breakouts for trend capture, volume to confirm breakout strength,
# and 1d ATR-based volatility filter to avoid low-momentum environments.
# Target: 80-180 total trades over 4 years (20-45/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) on 1d for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current / 20-period average) to detect volatility expansion
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / (atr_ma_20 + 1e-10)
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + volatility expansion (ATR ratio > 1.2)
        if (close[i] > donch_high_4h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            atr_ratio_aligned[i] > 1.2 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + volatility expansion (ATR ratio > 1.2)
        elif (close[i] < donch_low_4h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              atr_ratio_aligned[i] > 1.2 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or volatility contraction (ATR ratio < 0.8)
        elif position == 1 and (close[i] < donch_low_4h_aligned[i] or atr_ratio_aligned[i] < 0.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_4h_aligned[i] or atr_ratio_aligned[i] < 0.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Volume_Volatility_Filter"
timeframe = "4h"
leverage = 1.0