#!/usr/bin/env python3
# 4h_12h_trix_volume_regime_v1
# Strategy: 4h TRIX momentum with volume confirmation and 12h chop regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: TRIX captures momentum shifts, volume confirms strength, and 12h chop regime avoids false signals in sideways markets. Works in bull via TRIX>0 & rising, in bear via TRIX<0 & falling. Low trade frequency to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_trix_volume_regime_v1"
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
    
    # Load 12h data ONCE before loop for chop regime filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h Choppy Index (CHOP) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([high_12h[0] - low_12h[0]]), tr])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppy Index: 100 * log(sum(ATR,14) / (max(high,14) - min(low,14))) / log(14)
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    chop_regime = chop < 50  # <50 = trending, >50 = choppy
    chop_regime_aligned = align_htf_to_ltf(prices, df_12h, chop_regime)
    
    # 4h TRIX (12,9,9)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # 4h TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX momentum: above/below signal line
        trix_bullish = trix[i] > trix_signal[i]
        trix_bearish = trix[i] < trix_signal[i]
        
        # Regime filter: only trade in trending markets (CHOP < 50)
        trending = chop_regime_aligned[i]
        
        # Entry logic: TRIX cross + volume + trending regime
        if trix_bullish and vol_confirm[i] and trending and position != 1:
            position = 1
            signals[i] = 0.25
        elif trix_bearish and vol_confirm[i] and trending and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite TRIX signal
        elif position == 1 and trix_bearish:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix_bullish:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals