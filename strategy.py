#!/usr/bin/env python3
"""
4h_Pivot_Reversal_Strategy
Hypothesis: In ranging markets, price tends to reverse at daily pivot points (support/resistance).
We combine daily pivot levels with 4-hour RSI extremes and volume confirmation to capture mean-reversion bounces.
Works in both bull and bear markets as it fades extremes rather than following trend.
Target: 20-30 trades per year (~80-120 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_Reversal_Strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    # Support 2: S2 = P - (H - L)
    # Resistance 2: R2 = P + (H - L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = 2 * pivot - high_1d
    r1 = 2 * pivot - low_1d
    s2 = pivot - (high_1d - low_1d)
    r2 = pivot + (high_1d - low_1d)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # 4-hour RSI(14) for overbought/oversold conditions
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient data for RSI and volume
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long setup: price near S1 or S2 + RSI oversold (<30) + volume
            near_support = (abs(close[i] - s1_aligned[i]) < 0.005 * close[i]) or (abs(close[i] - s2_aligned[i]) < 0.005 * close[i])
            rsi_oversold = rsi[i] < 30
            
            # Short setup: price near R1 or R2 + RSI overbought (>70) + volume
            near_resistance = (abs(close[i] - r1_aligned[i]) < 0.005 * close[i]) or (abs(close[i] - r2_aligned[i]) < 0.005 * close[i])
            rsi_overbought = rsi[i] > 70
            
            if near_support and rsi_oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif near_resistance and rsi_overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches pivot point or RSI becomes overbought
            if close[i] >= pivot_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches pivot point or RSI becomes oversold
            if close[i] <= pivot_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals