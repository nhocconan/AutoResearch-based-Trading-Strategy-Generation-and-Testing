#!/usr/bin/env python3
"""
4h_1d_Trix_VolumeSpike_ChopRegime
Hypothesis: 4-hour TRIX momentum with volume spike confirmation and 1-day chop regime filter.
Long when TRIX crosses above zero with volume spike in trending market (CHOP < 38.2).
Short when TRIX crosses below zero with volume spike in trending market.
Works in both bull and bear markets via regime filter that avoids whipsaw in ranging conditions.
"""

name = "4h_1d_Trix_VolumeSpike_ChopRegime"
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
    
    # TRIX on close (4h): EMA(EMA(EMA(close,12),12),12) - 1 period ROC
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # percentage change
    trix_values = trix.values
    
    # Volume spike: >1.8x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d data for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Chopping index: 100 * log10(SUM(ATR,14) / (HHV - LLV)) / log10(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    hhvl = df_1d['high'].rolling(window=14, min_periods=14).max()
    llvl = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * (np.log10(atr.rolling(window=14, min_periods=14).sum()) - 
                  np.log10(hhvl - llvl)) / np.log10(14)
    chop_values = chop.values
    
    # Trending regime: CHOP < 38.2
    trending_regime = chop_values < 38.2
    
    # Align 1d data to 4h timeframe
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(trix_values[i]) or 
            np.isnan(trending_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero + volume spike + trending regime
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                volume_spike[i] and 
                trending_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike + trending regime
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  volume_spike[i] and 
                  trending_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR chop regime shifts to ranging
            if (trix_values[i] < 0 and trix_values[i-1] >= 0) or \
               not trending_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR chop regime shifts to ranging
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) or \
               not trending_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals