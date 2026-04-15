#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h EMA crossover + volume confirmation + volatility filter
# Uses EMA crossover for trend changes, volume to confirm momentum,
# and volatility filter (ATR ratio) to avoid choppy markets.
# Works in both bull and bear by trading EMA crossovers with volume confirmation.
# Target: 60-150 total trades over 4 years (15-38/year) with selective entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for ATR calculation (volatility filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA12 and EMA26 on 4h for crossover
    ema12_4h = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26_4h = pd.Series(close_4h).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate ATR (14-period) on 12h for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_avg_50 = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_12h / (atr_avg_50 + 1e-10)
    
    # Volume average (20-period on 4h)
    vol_avg_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema12_4h_aligned = align_htf_to_ltf(prices, df_4h, ema12_4h)
    ema26_4h_aligned = align_htf_to_ltf(prices, df_4h, ema26_4h)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema12_4h_aligned[i]) or np.isnan(ema26_4h_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: EMA12 crosses above EMA26 + volume spike + low volatility (atr_ratio < 1.5)
        if (ema12_4h_aligned[i] > ema26_4h_aligned[i] and
            ema12_4h_aligned[i-1] <= ema26_4h_aligned[i-1] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            atr_ratio_aligned[i] < 1.5 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: EMA12 crosses below EMA26 + volume spike + low volatility (atr_ratio < 1.5)
        elif (ema12_4h_aligned[i] < ema26_4h_aligned[i] and
              ema12_4h_aligned[i-1] >= ema26_4h_aligned[i-1] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              atr_ratio_aligned[i] < 1.5 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or high volatility (atr_ratio > 2.0) to avoid chop
        elif position == 1 and (ema12_4h_aligned[i] < ema26_4h_aligned[i] or atr_ratio_aligned[i] > 2.0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (ema12_4h_aligned[i] > ema26_4h_aligned[i] or atr_ratio_aligned[i] > 2.0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_EMA_Crossover_Volume_Volatility_Filter"
timeframe = "4h"
leverage = 1.0