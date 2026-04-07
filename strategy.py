#!/usr/bin/env python3
"""
6h Ehlers Fisher Transform with 1d Regime Filter
Long when Fisher crosses above -1.5 and 1d EMA50 > EMA200 (bullish regime)
Short when Fisher crosses below +1.5 and 1d EMA50 < EMA200 (bearish regime)
Exit when Fisher crosses back through zero
Designed to capture reversals in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_fisher_transform_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Ehlers Fisher Transform (9) ===
    price = (high + low) / 2
    # Normalize price to [-1, 1] range
    max_h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_l = pd.Series(low).rolling(window=9, min_periods=9).min().values
    diff = max_h - min_l
    diff = np.where(diff == 0, 1, diff)  # Avoid division by zero
    value1 = 2 * ((price - min_l) / diff - 0.5)
    # Smooth
    value1 = pd.Series(value1).ewm(alpha=0.5, adjust=False).mean().values
    # Fisher transform
    value1 = np.clip(value1, -0.999, 0.999)  # Avoid log(0)
    fish = 0.5 * np.log((1 + value1) / (1 - value1))
    fish = pd.Series(fish).ewm(alpha=0.5, adjust=False).mean().values
    
    # === 1d EMA Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        if np.isnan(fish[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Fisher crosses below zero
            if fish[i] < 0 and fish[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Fisher crosses above zero
            if fish[i] > 0 and fish[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish regime: EMA50 > EMA200
            # Bearish regime: EMA50 < EMA200
            if ema_50_aligned[i] > ema_200_aligned[i]:
                # Bullish regime - look for long
                if fish[i] > -1.5 and fish[i-1] <= -1.5:
                    position = 1
                    signals[i] = 0.25
            elif ema_50_aligned[i] < ema_200_aligned[i]:
                # Bearish regime - look for short
                if fish[i] < 1.5 and fish[i-1] >= 1.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals