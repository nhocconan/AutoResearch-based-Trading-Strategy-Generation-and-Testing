#!/usr/bin/env python3
"""
4h_1d_trix_volume_regime_v1
Strategy: 4h TRIX with volume confirmation and 1d Choppiness regime filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: TRIX (12-period) captures momentum shifts in 4h timeframe. 
Volume confirmation (>1.5x average volume) filters false signals.
1d Choppiness Index (14-period) defines regime: CHOP > 61.8 = range (mean reversion), 
CHOP < 38.2 = trending (momentum). In trending regime, we take TRIX signals.
In ranging regime, we fade extreme TRIX values. This dual approach adapts to 
both bull/bear markets by focusing on momentum in trends and mean reversion in ranges.
Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX calculation (12-period)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Prepend NaN for alignment
    trix = np.concatenate([[np.nan], trix])
    
    # 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(tr_sum / (atr * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filters
        trending = chop_aligned[i] < 38.2   # Trending regime
        ranging = chop_aligned[i] > 61.8    # Ranging regime
        
        # TRIX signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        trix_extreme_up = trix[i] > 0.5     # Overbought
        trix_extreme_down = trix[i] < -0.5  # Oversold
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Trading logic
        if trending and vol_confirmed:
            # In trending regime, follow TRIX momentum
            if trix_cross_up and position != 1:
                position = 1
                signals[i] = 0.25
            elif trix_cross_down and position != -1:
                position = -1
                signals[i] = -0.25
        elif ranging and vol_confirmed:
            # In ranging regime, fade extreme TRIX values
            if trix_extreme_down and position != 1:
                position = 1
                signals[i] = 0.25
            elif trix_extreme_up and position != -1:
                position = -1
                signals[i] = -0.25
        
        # Exit conditions
        exit_long = position == 1 and (
            (trending and trix_cross_down) or  # In trend, exit on reverse signal
            (ranging and trix[i] >= 0)         # In range, exit when TRIX normalizes
        )
        exit_short = position == -1 and (
            (trending and trix_cross_up) or    # In trend, exit on reverse signal
            (ranging and trix[i] <= 0)         # In range, exit when TRIX normalizes
        )
        
        if exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals