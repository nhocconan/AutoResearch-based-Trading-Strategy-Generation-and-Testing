#!/usr/bin/env python3
"""
6h_volume_regime_follow_v1
Hypothesis: 6h trends are stronger when volume confirms price direction.
Use 1d ATR regime filter to distinguish trending vs ranging markets.
- In trending regime (ATR rising): follow 6h EMA crossover with volume confirmation
- In ranging regime (ATR falling): mean revert at Bollinger Bands
- Volume confirmation: current volume > 1.5 * 20-period average
- This filters false breakouts and improves win rate in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_regime_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h indicators
    # EMA crossover for trend
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).values
    
    # Bollinger Bands for mean reversion
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d ATR for regime detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR trend: rising if current > previous
    atr_rising = np.concatenate([[False], atr_14[1:] > atr_14[:-1]])
    
    # Align 1d indicators to 6h
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising.astype(float))
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_avg_20[i]) or np.isnan(atr_rising_aligned[i])):
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend following regime (ATR rising)
        if atr_rising_aligned[i] > 0.5:
            # EMA crossover with volume
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and vol_confirm:
                signals[i] = 0.25
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and vol_confirm:
                signals[i] = -0.25
        
        # Mean reversion regime (ATR falling)
        else:
            # Buy at lower band, sell at upper band
            if close[i] <= bb_lower[i] and close[i-1] > bb_lower[i-1]:
                signals[i] = 0.25
            elif close[i] >= bb_upper[i] and close[i-1] < bb_upper[i-1]:
                signals[i] = -0.25
    
    return signals