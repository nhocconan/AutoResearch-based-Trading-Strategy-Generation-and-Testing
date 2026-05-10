#!/usr/bin/env python3
"""
4h_Adaptive_Keltner_MeanReversion_TrendFilter
Hypothesis: Mean reversion at Keltner Channel (ATR-based) extremes with trend filter on 12h EMA200.
Buys near lower band in uptrend, sells near upper band in downtrend. Uses volume confirmation to avoid false signals.
Designed to work in both bull and bear markets by following 12h trend and fading extremes only when aligned with higher timeframe.
Target: 20-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "4h_Adaptive_Keltner_MeanReversion_TrendFilter"
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
    
    # Calculate ATR(20) for Keltner Channels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(20, n):
        if i == 20:
            atr[i] = np.mean(tr[1:21])
        else:
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Calculate EMA(20) for middle band
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    # Keltner Channels: EMA20 ± 1.5 * ATR(20)
    kc_upper = ema20 + 1.5 * atr
    kc_lower = ema20 - 1.5 * atr
    
    # Calculate 12h EMA200 for trend filter (using HTF data)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema200_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 200:
        ema200_12h[199] = np.mean(close_12h[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_12h)):
            ema200_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema200_12h[i-1]
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure EMA20, ATR, and 12h EMA200 are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(ema200_12h_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Price near lower Keltner band in uptrend (12h EMA200 up)
            if close[i] <= kc_lower[i] and close[i] > ema200_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price near upper Keltner band in downtrend (12h EMA200 down)
            elif close[i] >= kc_upper[i] and close[i] < ema200_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses back above EMA20 (mean reversion complete) or trend breaks
            if close[i] >= ema20[i] or close[i] < ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses back below EMA20 (mean reversion complete) or trend breaks
            if close[i] <= ema20[i] or close[i] > ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals