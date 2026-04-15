#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + daily volatility filter
# Uses Donchian channel breakouts on 12h timeframe for trend capture, volume to confirm breakout strength,
# and daily ATR ratio to avoid low volatility environments. Works in both bull and bear by
# only taking breakouts when volatility is expanding (ATR(7)/ATR(30) > 1.2).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for volatility filter (7 and 30 periods) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_7 / (atr_30 + 1e-10)  # Avoid division by zero
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + expanding volatility (ATR ratio > 1.2)
        if (close[i] > donch_high_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            atr_ratio_aligned[i] > 1.2 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + expanding volatility (ATR ratio > 1.2)
        elif (close[i] < donch_low_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              atr_ratio_aligned[i] > 1.2 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or low volatility (ATR ratio < 1.0)
        elif position == 1 and (close[i] < donch_low_12h_aligned[i] or atr_ratio_aligned[i] < 1.0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_12h_aligned[i] or atr_ratio_aligned[i] < 1.0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Volume_Volatility_Filter"
timeframe = "12h"
leverage = 1.0