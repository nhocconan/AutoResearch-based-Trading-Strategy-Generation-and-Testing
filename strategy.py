# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_Adaptive_Kelly_Volume_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d True Range and ATR(14)
    tr = np.maximum(df_1d['high'].values - df_1d['low'].values,
                    np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                               np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d RSI(14) for momentum filter
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h volume filter: > 1.8x 20-period average (adaptive to regime)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.8 * vol_ma_6h
    
    # 6h momentum: ROC(6) > 0 for long, < 0 for short
    roc_6h = np.zeros_like(close)
    roc_6h[6:] = (close[6:] - close[:-6]) / close[:-6]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_6h[i]) or np.isnan(roc_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Adaptive Kelly sizing based on volatility and momentum
        # Base size 0.25, scaled by volatility (inverse ATR) and momentum strength
        vol_factor = np.clip(0.5 * (atr_1d[0] / atr_1d_aligned[i]), 0.5, 1.5) if atr_1d_aligned[i] > 0 else 1.0
        mom_strength = np.clip(abs(roc_6h[i]) * 20, 0.5, 2.0)  # Scale ROC to reasonable range
        base_size = 0.25
        size = base_size * vol_factor * mom_strength
        size = np.clip(size, 0.15, 0.35)  # Keep within reasonable bounds
        
        if position == 0:
            # Long: 1d ATR expansion + RSI > 50 + 6m ROC positive + volume surge
            if (atr_1d_aligned[i] > atr_1d_aligned[i-1] * 1.1 and  # ATR expanding
                rsi_1d_aligned[i] > 50 and 
                roc_6h[i] > 0 and 
                vol_filter[i]):
                signals[i] = size
                position = 1
            # Short: 1d ATR expansion + RSI < 50 + 6m ROC negative + volume surge
            elif (atr_1d_aligned[i] > atr_1d_aligned[i-1] * 1.1 and  # ATR expanding
                  rsi_1d_aligned[i] < 50 and 
                  roc_6h[i] < 0 and 
                  vol_filter[i]):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: ATR contraction or RSI < 40 or ROC turns negative
            if (atr_1d_aligned[i] < atr_1d_aligned[i-1] * 0.9 or 
                rsi_1d_aligned[i] < 40 or 
                roc_6h[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: ATR contraction or RSI > 60 or ROC turns positive
            if (atr_1d_aligned[i] < atr_1d_aligned[i-1] * 0.9 or 
                rsi_1d_aligned[i] > 60 or 
                roc_6h[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

# Hypothesis: 6s Adaptive Kelly Volume Momentum
# Uses 1d ATR expansion to detect volatility regimes (works in both bull/bear markets)
# Combined with 1d RSI for directional bias and 6h ROC for momentum confirmation
# Volume surge filter ensures institutional participation
# Adaptive Kelly sizing reduces exposure in high volatility, increases in low volatility
# Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag
# Position sizing adapts to market conditions for better risk-adjusted returns