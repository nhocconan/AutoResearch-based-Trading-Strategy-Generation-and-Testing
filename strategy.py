#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX momentum with 1d volume spike and 1w choppiness regime filter.
- TRIX (12,9,9) captures smooth momentum - long when TRIX rises above zero line, short when falls below zero.
- Volume confirmation: 12h volume > 1.5x 20-bar average to avoid low-volume false signals.
- Regime filter: 1w choppiness index > 61.8 (ranging market) enables mean-reversion exits at extremes.
- Designed for 12h timeframe to work in both bull (trend following) and bear (mean reversion in chop) markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Novelty: Combines TRIX momentum with weekly chop regime for adaptive behavior in changing markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for choppiness index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on 1d close: EMA(EMA(EMA(close,12),9),9)
    ema1 = pd.Series(df_1d['close']).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_raw = 100 * (ema3.pct_change())
    trix = trix_raw.values
    
    # Align TRIX to 12h timeframe (wait for 1d bar to close)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate 1w choppiness index: CHOP = 100 * log10(sum(ATR(14),14) / (max(high,14)-min(low,14))) / log10(14)
    tr_1w = np.maximum(
        df_1w['high'].values - df_1w['low'].values,
        np.maximum(
            np.abs(df_1w['high'].values - np.concatenate([[df_1w['close'].values[0]], df_1w['close'].values[:-1]])),
            np.abs(df_1w['low'].values - np.concatenate([[df_1w['close'].values[0]], df_1w['close'].values[:-1]]))
        )
    )
    atr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean()
    max_high_14 = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max()
    min_low_14 = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min()
    chop_raw = 100 * np.log10(atr_14.rolling(window=14, min_periods=14).sum() / (max_high_14 - min_low_14)) / np.log10(14)
    chop = chop_raw.values
    
    # Align choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume confirmation: > 1.5x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # Need enough for TRIX and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Long: TRIX crosses above zero AND chop < 61.8 (trending) OR chop > 61.8 with mean reversion bias
                if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                    # In trending regime (chop < 61.8): follow TRIX momentum
                    # In choppy regime (chop > 61.8): only long if TRIX deeply negative (oversold)
                    if chop_aligned[i] < 61.8 or trix_aligned[i] < -0.1:
                        signals[i] = 0.25
                        position = 1
                # Short: TRIX crosses below zero AND chop < 61.8 (trending) OR chop > 61.8 with mean reversion bias
                elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                    # In trending regime (chop < 61.8): follow TRIX momentum
                    # In choppy regime (chop > 61.8): only short if TRIX deeply positive (overbought)
                    if chop_aligned[i] < 61.8 or trix_aligned[i] > 0.1:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR chop > 61.8 with TRIX near zero (chop reversion)
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and abs(trix_aligned[i]) < 0.05:  # Chop regime mean reversion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero OR chop > 61.8 with TRIX near zero (chop reversion)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and abs(trix_aligned[i]) < 0.05:  # Chop regime mean reversion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_1wChopRegime_v1"
timeframe = "12h"
leverage = 1.0