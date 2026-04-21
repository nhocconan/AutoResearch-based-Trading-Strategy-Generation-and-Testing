#!/usr/bin/env python3
"""
1h_HTF_Momentum_LowVol
Hypothesis: Use 4h and 1d momentum (ROC) to establish trend direction, enter on 1h pullbacks during low volatility periods (low ATR ratio). Exit when momentum fades or volatility expands. Designed to capture trend continuation moves with low slippage in both bull and bear markets by aligning with higher timeframe momentum while using volatility filter to avoid choppy periods. Target 20-40 trades/year on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h momentum: ROC(10) ===
    close_4h = df_4h['close'].values
    roc_4h = np.zeros_like(close_4h)
    roc_4h[10:] = (close_4h[10:] - close_4h[:-10]) / close_4h[:-10] * 100
    roc_4h_smoothed = pd.Series(roc_4h).ewm(span=5, adjust=False, min_periods=5).mean().values
    roc_4h_aligned = align_htf_to_ltf(prices, df_4h, roc_4h_smoothed)
    
    # === 1d momentum: ROC(5) ===
    close_1d = df_1d['close'].values
    roc_1d = np.zeros_like(close_1d)
    roc_1d[5:] = (close_1d[5:] - close_1d[:-5]) / close_1d[:-5] * 100
    roc_1d_smoothed = pd.Series(roc_1d).ewm(span=3, adjust=False, min_periods=3).mean().values
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d_smoothed)
    
    # === 1h volatility filter: ATR ratio (current ATR / 24-period average ATR) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_24 = pd.Series(atr).rolling(window=24, min_periods=24).mean().values
    atr_ratio = np.where(atr_ma_24 > 0, atr / atr_ma_24, 1.0)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(roc_4h_aligned[i]) or
            np.isnan(roc_1d_aligned[i]) or
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        mom_4h = roc_4h_aligned[i]
        mom_1d = roc_1d_aligned[i]
        vol_filter = atr_ratio[i]
        
        if position == 0:
            # Long: strong 4h and 1d momentum (+) + low volatility (mean reversion setup)
            if mom_4h > 0.5 and mom_1d > 0.3 and vol_filter < 0.8:
                signals[i] = 0.20
                position = 1
            # Short: strong 4h and 1d momentum (-) + low volatility
            elif mom_4h < -0.5 and mom_1d < -0.3 and vol_filter < 0.8:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: momentum fades or volatility expands
            if position == 1:
                if mom_4h < 0.2 or mom_1d < 0.1 or vol_filter > 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if mom_4h > -0.2 or mom_1d > -0.1 or vol_filter > 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_HTF_Momentum_LowVol"
timeframe = "1h"
leverage = 1.0