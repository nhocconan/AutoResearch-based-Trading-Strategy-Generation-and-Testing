#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_RegimeFilter
Hypothesis: Camarilla R3/S3 breakouts from 1d HTF with volume confirmation (top 30%) and choppiness regime filter (CHOP > 61.8 = ranging). Uses discrete position sizing (0.25) to limit trades. Designed for ranging markets where mean reversion to midpoint works. Target: 20-30 trades/year to avoid fee drag.
"""

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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime = volume > vol_percentile_70
    
    # Choppiness regime: CHOP > 61.8 = ranging market (good for mean reversion)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.log(highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_regime = chop > 61.8  # ranging market
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and volume percentile, 14 for ATR)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_percentile_70[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_regime = volume_regime[i]
        chop_reg = chop_regime[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume regime AND chop regime (ranging)
        long_entry = (close_val > camarilla_r3_val) and vol_regime and chop_reg
        short_entry = (close_val < camarilla_s3_val) and vol_regime and chop_reg
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_RegimeFilter"
timeframe = "4h"
leverage = 1.0