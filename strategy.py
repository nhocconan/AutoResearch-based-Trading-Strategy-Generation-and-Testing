#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_Regime
Hypothesis: Trade 12h timeframe using TRIX (15) momentum for entry, confirmed by daily volume spike (>2.0x 20-bar MA) and choppiness regime filter (CHOP(14) > 61.8 for ranging markets). 
Enter long when TRIX crosses above zero AND volume spike AND choppy regime. 
Enter short when TRIX crosses below zero AND volume spike AND choppy regime. 
Exit on opposite TRIX cross. Uses discrete sizing 0.25 to balance return and drawdown. 
Targets 12-37 trades/year on 12h timeframe. Works in bull/bear via momentum + regime filter avoiding strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for TRIX, volume spike, and choppiness
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX (15) on daily close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = 0
    trix_cross_above = (trix > 0) & (trix_prev <= 0)
    trix_cross_below = (trix < 0) & (trix_prev >= 0)
    
    # Align TRIX and cross signals to 12h timeframe (completed daily bar only)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_cross_above_aligned = align_htf_to_ltf(prices, df_1d, trix_cross_above.astype(float))
    trix_cross_below_aligned = align_htf_to_ltf(prices, df_1d, trix_cross_below.astype(float))
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Choppiness Index (14) on 1d
    # CHOP = 100 * log10(sum(ATR(1), 14) / (max(high,14) - min(low,14))) / log10(14)
    atr_1 = np.maximum(high_1d - low_1d, np.maximum(abs(high_1d - np.roll(close_1d, 1)), abs(low_1d - np.roll(close_1d, 1))))
    atr_1[0] = high_1d[0] - low_1d[0]
    sum_atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr_14) - np.log10(max_high_14 - min_low_14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_regime = chop > 61.8  # ranging market
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for TRIX (45), volume MA (20), chop (14)
    start_idx = max(45, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_cross_above_aligned[i]) or np.isnan(trix_cross_below_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero AND volume spike AND choppy regime
            long_setup = trix_cross_above_aligned[i] and volume_spike_1d_aligned[i] and chop_regime_aligned[i]
            # Short: TRIX crosses below zero AND volume spike AND choppy regime
            short_setup = trix_cross_below_aligned[i] and volume_spike_1d_aligned[i] and chop_regime_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TRIX crosses below zero
            if trix_cross_below_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX crosses above zero
            if trix_cross_above_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_TRIX_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0