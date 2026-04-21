#!/usr/bin/env python3
"""
6h_1d_1w_Adaptive_Kelly_Volume_Regime_v1
Hypothesis: Adaptive Kelly sizing based on volatility regime (6h ATR ratio) and 1d/1w trend alignment.
In high volatility regime (expanding ATR), reduce size; in low volatility (contraction), increase size.
Only trade when 1d and 1w EMA50 agree on direction. Uses volume confirmation for entry timing.
Target: 12-25 trades/year per symbol. Works in bull/bear by aligning with higher timeframe trend and adapting to volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data once for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h ATR for volatility regime
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma + 1e-10)
    
    # Base Kelly fraction (adjusted for win rate ~0.55, avg win/loss ~1.2)
    kelly_base = 0.25  # conservative base
    
    # Volatility scaling: inverse relationship with ATR ratio
    # When ATR ratio > 1.5 (high vol), scale down; < 0.8 (low vol), scale up
    vol_scale = np.clip(1.0 / atr_ratio, 0.5, 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Determine trend direction from 1d and 1w EMA50
        trend_1d = 1 if price > ema_50_1d_aligned[i] else -1
        trend_1w = 1 if price > ema_50_1w_aligned[i] else -1
        
        # Only trade when both timeframes agree
        if trend_1d == trend_1w and volume_ok:
            # Adaptive position size based on volatility regime
            size = kelly_base * vol_scale[i]
            size = np.clip(size, 0.15, 0.35)  # enforce limits
            
            if trend_1d == 1 and position <= 0:  # go long
                signals[i] = size
                position = 1
            elif trend_1d == -1 and position >= 0:  # go short
                signals[i] = -size
                position = -1
            else:
                # Hold current position
                signals[i] = signals[i-1] if i > 0 else 0.0
        else:
            # No clear signal or volume filter failed - exit or hold flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_Adaptive_Kelly_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0