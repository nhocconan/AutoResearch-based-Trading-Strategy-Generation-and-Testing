#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_V1
Hypothesis: TRIX momentum with volume confirmation and choppiness regime filter works on 4h timeframe for BTC and ETH in both bull and bear markets.
- TRIX(12) captures smoothed momentum with reduced whipsaw
- Volume spike (>2x 20-bar average) confirms institutional participation
- Choppiness Index (14) > 61.8 defines ranging markets for mean-reversion TRIX signals
- Discrete position sizing (0.25) minimizes fee churn
- Target: 25-60 trades/year per symbol (100-240 over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for higher timeframe regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX(12) - Triple Exponential Average momentum
    close = prices['close'].values
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = np.nan  # First value is invalid
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14) from 1d timeframe for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = np.abs(np.diff(low_1d, prepend=np.nan))
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    atr_period = 14
    tr_period = pd.Series(atr_1d).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / tr_period
    minus_di = 100 * pd.Series(minus_dm).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values / tr_period
    
    # DX and Choppiness Index
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(plus_di) | np.isnan(minus_di) | (plus_di + minus_di) == 0] = np.nan
    chop = 100 * np.log10(pd.Series(tr).rolling(window=14, min_periods=14).sum().values) / np.log10(14)
    
    # Align 1d chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>2x average)
        volume_ok = volume > 2.0 * vol_ma[i]
        
        # Choppiness regime: > 61.8 = ranging market (mean reversion favorable)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: TRIX crosses above zero in ranging market with volume
            if trix[i] > 0 and np.roll(trix, 1)[i] <= 0 and chop_regime and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero in ranging market with volume
            elif trix[i] < 0 and np.roll(trix, 1)[i] >= 0 and chop_regime and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero or volume dries up
            if trix[i] < 0 or volume < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above zero or volume dries up
            if trix[i] > 0 or volume < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_V1"
timeframe = "4h"
leverage = 1.0