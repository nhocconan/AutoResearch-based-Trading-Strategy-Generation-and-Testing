#!/usr/bin/env python3
# 4h_TripleConfirmation_Breakout_Strategy
# Hypothesis: Combine 4h Donchian breakout with 12h volume spike and 1d volatility regime filter to create high-probability entries.
# Uses Donchian(20) breakouts confirmed by volume spikes and filtered by low-volatility (range-bound) conditions.
# Works in both bull and bear markets by focusing on breakouts from consolidation regardless of trend direction.
# Target: 25-40 trades/year (~100-160 total over 4 years) to minimize fee drag.

name = "4h_TripleConfirmation_Breakout_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volatility regime filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume moving average for spike detection
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 1d volatility regime using ATR percentage of price
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14) and ATR as percentage of price
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_percent = (atr_1d / close_1d) * 100
    
    # Define low volatility regime: ATR percentage below 30th percentile
    # Using rolling percentile for adaptive threshold
    atr_percent_series = pd.Series(atr_percent)
    atr_percent_30th = atr_percent_series.rolling(window=50, min_periods=30).quantile(0.30).values
    low_vol_regime = atr_percent < atr_percent_30th
    
    # Align volatility regime to 4h timeframe
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(low_vol_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Donchian breakout + volume spike + low volatility regime
            if (close[i] > highest_high[i] and 
                volume[i] > 2.0 * vol_ma_12h_aligned[i] and
                low_vol_aligned[i] > 0.5):  # In low volatility regime
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakdown + volume spike + low volatility regime
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 2.0 * vol_ma_12h_aligned[i] and
                  low_vol_aligned[i] > 0.5):  # In low volatility regime
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakdown or volatility regime change
            if (close[i] < lowest_low[i] or 
                low_vol_aligned[i] <= 0.5):  # Exit low volatility regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout or volatility regime change
            if (close[i] > highest_high[i] or 
                low_vol_aligned[i] <= 0.5):  # Exit low volatility regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals