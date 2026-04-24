#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX Momentum with 1d Volume Spike and Choppiness Regime Filter.
- TRIX (12) identifies momentum shifts with reduced lag vs MACD.
- 1d volume spike (>2.0x 20-period average) confirms institutional participation.
- 1d Choppiness Index > 61.8 indicates ranging market (mean reversion); < 38.2 indicates trending.
- In trending regime (CHOP < 38.2): TRIX crosses above/below zero line for trend continuation.
- In ranging regime (CHOP > 61.8): TRIX extremes (>0.10 long, <-0.10 short) for mean reversion.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag.
- Works in bull/bear markets via regime adaptation and volume confirmation.
"""

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
    
    # Get 1d data ONCE before loop for TRIX, volume, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA for TRIX calculation (triple EMA)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value is invalid due to roll
    
    # 1d volume spike confirmation
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > 2.0 * vol_ma
    
    # 1d Choppiness Index (CHOP) - measures ranging vs trending
    atr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            atr_1d.append(0)
        else:
            tr = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
            atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    high_max = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    low_min = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((high_max - low_min) > 0, chop, 50.0)
    
    # Align all 1d indicators to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 12*3) + 1  # volume(20), chop(14), TRIX needs ~36 bars
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike_aligned[i] > 0.5  # boolean as float
        
        if position == 0:
            if vol_confirm:
                # Regime-based entry
                if chop_aligned[i] < 38.2:  # Trending regime
                    # Long: TRIX crosses above zero with momentum
                    if trix_aligned[i] > 0 and trix_aligned[i] > trix_aligned[i-1]:
                        signals[i] = 0.25
                        position = 1
                    # Short: TRIX crosses below zero with momentum
                    elif trix_aligned[i] < 0 and trix_aligned[i] < trix_aligned[i-1]:
                        signals[i] = -0.25
                        position = -1
                elif chop_aligned[i] > 61.8:  # Ranging regime
                    # Long: TRIX oversold mean reversion
                    if trix_aligned[i] < -0.10 and trix_aligned[i] > trix_aligned[i-1]:
                        signals[i] = 0.25
                        position = 1
                    # Short: TRIX overbought mean reversion
                    elif trix_aligned[i] > 0.10 and trix_aligned[i] < trix_aligned[i-1]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: TRIX momentum loss or regime change to extreme ranging
            if trix_aligned[i] < 0 or chop_aligned[i] > 70.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX momentum loss or regime change to extreme ranging
            if trix_aligned[i] > 0 or chop_aligned[i] > 70.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopperRegime_v1"
timeframe = "12h"
leverage = 1.0