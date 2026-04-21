#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChopRegime_V1
Hypothesis: TRIX momentum with volume spike confirmation and choppiness regime filter works on 12h timeframe for BTC and ETH in both bull and bear markets. Uses 1d HTF for chop regime and 1w HTF for trend context. Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data once for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # TRIX calculation (15-period EMA triple)
    close = prices['close'].values
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = np.nan
    trix_sig = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Choppiness Index (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            atr_1d.append(np.nan)
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
            atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop[np.isnan(highest_high - lowest_low) | (highest_high - lowest_low) == 0] = np.nan
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
        if (np.isnan(trix_sig[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Regime filter: chop < 61.8 = trending (use TRIX), chop > 61.8 = ranging (avoid)
        trending_regime = chop_aligned[i] < 61.8
        
        # 1w trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: TRIX crosses above signal in uptrend + volume + trending regime
            if uptrend and volume_ok and trending_regime:
                if trix[i] > trix_sig[i] and trix[i-1] <= trix_sig[i-1]:
                    signals[i] = 0.30
                    position = 1
            # Short: TRIX crosses below signal in downtrend + volume + trending regime
            elif downtrend and volume_ok and trending_regime:
                if trix[i] < trix_sig[i] and trix[i-1] >= trix_sig[i-1]:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below signal or stoploss
            if trix[i] < trix_sig[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: TRIX crosses above signal or stoploss
            if trix[i] > trix_sig[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_V1"
timeframe = "12h"
leverage = 1.0