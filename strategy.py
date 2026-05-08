#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX + volume spike + choppiness regime filter for trend strength.
# Long when TRIX crosses above zero AND choppy market (CHOP > 61.8) AND volume > 1.5x 20-period average.
# Short when TRIX crosses below zero AND choppy market AND volume spike.
# Exit when TRIX crosses back in opposite direction or chop regime shifts to trending (CHOP < 38.2).
# TRIX captures momentum with less whipsaw, chop regime avoids false signals in strong trends,
# volume confirms institutional participation. Target: 50-150 total trades over 4 years.

name = "12h_TRIX_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for TRIX calculation and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX calculation (1-period EMA of 1-period EMA of 1-period EMA of close)
    ema1 = pd.Series(df_1d['close']).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # percentage change
    
    # Choppiness Index calculation
    atr_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_hh = df_1d['high'].rolling(window=atr_period, min_periods=atr_period).max().values
    min_ll = df_1d['low'].rolling(window=atr_period, min_periods=atr_period).min().values
    
    chop = np.zeros_like(atr)
    for i in range(atr_period, len(atr)):
        if max_hh[i] > min_ll[i]:
            chop[i] = 100 * np.log10(sum(tr[i-atr_period+1:i+1]) / (np.log10(atr_period) * (max_hh[i] - min_ll[i])))
        else:
            chop[i] = 50  # avoid division by zero
    
    # Align TRIX and chop to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 15)  # Sufficient warmup for TRIX and chop
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX cross signals
        trix_cross_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
        trix_cross_down = trix_aligned[i] < 0 and trix_aligned[i-1] >= 0
        
        # Chop regime: > 61.8 = ranging (good for mean reversion), < 38.2 = trending
        chop_ranging = chop_aligned[i] > 61.8
        chop_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long conditions: TRIX cross up, chop ranging, volume filter
            long_cond = trix_cross_up and chop_ranging and volume_filter[i]
            # Short conditions: TRIX cross down, chop ranging, volume filter
            short_cond = trix_cross_down and chop_ranging and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX cross down OR chop becomes trending
            if trix_cross_down or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX cross up OR chop becomes trending
            if trix_cross_up or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals