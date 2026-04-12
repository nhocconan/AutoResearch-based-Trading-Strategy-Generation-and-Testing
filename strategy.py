#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_trix_volume_regime
# Uses TRIX momentum on 4h with volume confirmation and daily Chop regime filter.
# Long when TRIX crosses above zero with volume > 1.5x average and Chop > 61.8 (range).
# Short when TRIX crosses below zero with volume > 1.5x average and Chop > 61.8.
# Exit when TRIX returns to zero (mean reversion in ranging markets).
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
# TRIX filters noise, volume confirms breakout, Chop ensures mean-reversion context.

name = "4h_1d_trix_volume_regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Chop index (daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = highest_high - lowest_low
    chop = np.where(denominator > 0, 100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX on 4h close (1-period EMA of 1-period EMA of 1-period EMA, then ROC)
    # EMA1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (EMA3 - prev EMA3) / prev EMA3
    trix = np.zeros_like(close)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix[0] = 0
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(chop_aligned[i]) or np.isnan(trix[i]):
            signals[i] = 0.0
            continue
        
        # Range condition: Chop > 61.8 (ranging market)
        if chop_aligned[i] <= 61.8:
            # Hold current position if not in range
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: TRIX crosses above zero
        if trix[i] > 0 and trix[i-1] <= 0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: TRIX crosses below zero
        elif trix[i] < 0 and trix[i-1] >= 0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: TRIX returns to zero (mean reversion)
        elif position == 1 and trix[i] <= 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix[i] >= 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals