#!/usr/bin/env python3
"""
6h_12h_TRIX_ZeroCross_Volume_Regime_v1
Hypothesis: TRIX (triple smoothed EMA) zero-cross signals combined with volume confirmation and regime filter (Choppiness Index) to avoid whipsaws in sideways markets. TRIX captures momentum shifts while volume confirms institutional participation. Choppiness Index > 61.8 filters out ranging conditions where TRIX whipsaws. Targets 15-30 trades/year per symbol with focus on high-probability momentum shifts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_TRIX_ZeroCross_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for TRIX and Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate TRIX on 12h close (15,9,9 smoothing)
    close_12h = df_12h['close'].values
    # First EMA (15-period)
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA (9-period on first EMA)
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Third EMA (9-period on second EMA)
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX = % change of third EMA
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Calculate Choppiness Index on 12h data (14-period)
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Sum of absolute price change over 14 periods
    atr_sum = pd.Series(abs(df_12h['high'] - df_12h['low'])).rolling(window=14, min_periods=14).sum().values
    # Choppiness Index = 100 * log10(tr_sum / atr_sum) / log10(14)
    chop = np.full_like(tr_sum, 50.0, dtype=float)  # default neutral
    valid = (atr_sum > 0) & ~np.isnan(tr_sum) & ~np.isnan(atr_sum)
    chop[valid] = 100 * np.log10(tr_sum[valid] / atr_sum[valid]) / np.log10(14)
    
    # Align TRIX and Choppiness Index to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume filter: 20-period average on 6h timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Regime filter: only trade when market is trending (Choppiness < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # TRIX zero-cross signals
        trix_now = trix_aligned[i]
        trix_prev = trix_aligned[i-1]
        
        # Zero-cross signals: bullish when crossing above zero, bearish when crossing below zero
        bullish_cross = (trix_prev <= 0) and (trix_now > 0)
        bearish_cross = (trix_prev >= 0) and (trix_now < 0)
        
        # Entry conditions: TRIX zero-cross + volume + trending regime
        long_entry = bullish_cross and volume_filter and trending_regime
        short_entry = bearish_cross and volume_filter and trending_regime
        
        # Exit conditions: opposite TRIX zero-cross or loss of trending regime
        long_exit = bearish_cross or (not trending_regime)
        short_exit = bullish_cross or (not trending_regime)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals