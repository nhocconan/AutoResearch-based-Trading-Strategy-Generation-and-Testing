#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_RegimeFilter_V1
Hypothesis: TRIX (15-period) crossover with volume spike (>2x 20-bar MA) and chop regime filter (CHOP(14) > 61.8) works on 4h timeframe for BTC and ETH in both bull and bear markets. Uses 1d timeframe for chop calculation to avoid look-ahead. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for chop regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on primary timeframe (4h)
    close = prices['close'].values
    # TRIX: EMA(EMA(EMA(close, 15), 15), 15) - 1 period percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate chop regime on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Chop = 100 * log10(sum(TR14) / (max_high14 - min_low14)) / log10(14)
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(atr_14) - np.log10(max_high_14 - min_low_14)) / np.log10(14)
    chop[np.isnan(chop) | np.isinf(chop)] = 50  # neutral when invalid
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(trix_signal[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>2x average)
        volume_ok = volume > 2.0 * vol_ma[i]
        
        # Chop regime: range-bound market (CHOP > 61.8)
        chop_ok = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: TRIX crosses above signal line in choppy market with volume
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]:
                if chop_ok and volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: TRIX crosses below signal line in choppy market with volume
            elif trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]:
                if chop_ok and volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below signal line or stoploss
            if trix[i] < trix_signal[i] or price < prices['close'].iloc[i-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above signal line or stoploss
            if trix[i] > trix_signal[i] or price > prices['close'].iloc[i-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_RegimeFilter_V1"
timeframe = "4h"
leverage = 1.0