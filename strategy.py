#!/usr/bin/env python3
"""
4H_TRIX_Momentum_VolumeSpike
Hypothesis: 4h TRIX momentum combined with volume spikes and choppiness regime filter captures strong trends in both bull and bear markets. TRIX filters noise, volume confirms strength, chop filter avoids ranging markets. Targets 20-40 trades/year on 4h timeframe to minimize fee drag.
"""
name = "4H_TRIX_Momentum_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1D Choppiness Index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14) and sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(tr_sum / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop = np.where(atr_14 * 14 > 0, chop, 50)  # Avoid division by zero
    chop = np.where(np.isnan(chop), 50, chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX (15,9,9) on 4h close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    trix = trix.fillna(0).values
    
    # Volume filter: current 4h volume > 2.0 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(35, 20)  # Ensure sufficient warmup for TRIX and volume
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 4h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: TRIX turning up from negative, chop < 61.8 (trending), volume spike
            if (trix[i] > trix[i-1] and trix[i-1] <= 0 and 
                chop_aligned[i] < 61.8 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: TRIX turning down from positive, chop < 61.8 (trending), volume spike
            elif (trix[i] < trix[i-1] and trix[i-1] >= 0 and 
                  chop_aligned[i] < 61.8 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: TRIX crosses zero (momentum reversal)
            if position == 1 and trix[i] < 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and trix[i] > 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals