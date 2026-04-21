#!/usr/bin/env python3
"""
1d_Keltner_R1S1_Breakout_V1
Hypothesis: Keltner channels on 1d (ATR-based) capture volatility breakouts while R1/S1 pivots from 1w provide key support/resistance levels. Long when price breaks above Keltner upper band and above weekly R1; short when breaks below Keltner lower band and below weekly S1. Uses volume confirmation and ATR stop. Designed to work in both bull and bear by following volatility expansion with institutional pivot levels as filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R1, S1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivots to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate Keltner Channel on daily data (20-period EMA, 2x ATR)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(20) for middle band
    alpha = 2 / (20 + 1)
    ema = np.full_like(close, np.nan)
    ema[0] = close[0]
    for i in range(1, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    
    # ATR(20) for band width
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(close, np.nan)
    for i in range(20, len(tr)):
        if not np.isnan(tr[i]):
            atr[i] = np.mean(tr[i-19:i+1])
    
    # Keltner bands
    keltner_upper = ema + 2 * atr
    keltner_lower = ema - 2 * atr
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i] if not np.isnan(volume_filter[i]) else False
        
        if position == 0:
            # Long: price breaks above Keltner upper AND above weekly R1 with volume
            if price > keltner_upper[i] and price > r1_1w_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner lower AND below weekly S1 with volume
            elif price < keltner_lower[i] and price < s1_1w_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Keltner middle or below weekly S1
            if price < ema[i] or price < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Keltner middle or above weekly R1
            if price > ema[i] or price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_R1S1_Breakout_V1"
timeframe = "1d"
leverage = 1.0