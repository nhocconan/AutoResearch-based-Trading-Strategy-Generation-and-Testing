#!/usr/bin/env python3
"""
4h_1d_KAMA_Direction_With_Volume_Confirmation
Hypothesis: Use KAMA on 1d to determine trend direction (adaptive trend strength) and 4h for entry timing with volume confirmation.
Long when KAMA slope positive on 1d, price > KAMA on 4h, and volume > 1.5x 20-period average.
Short when KAMA slope negative on 1d, price < KAMA on 4h, and volume > 1.5x 20-period average.
Exit when price crosses back below/above KAMA on 4h.
Uses adaptive trend strength to capture trends while avoiding whipsaws in ranging markets.
Target: 20-40 trades/year per symbol. Works in bull/bear by following adaptive trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for KAMA trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 1d
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i-1]):
            kama_1d[i] = kama_1d[i-1] + sc[i-1] * (close_1d[i] - kama_1d[i-1])
        else:
            kama_1d[i] = kama_1d[i-1]
    
    # KAMA slope (trend direction)
    kama_slope_1d = np.diff(kama_1d, 1)
    kama_slope_1d = np.concatenate([[np.nan], kama_slope_1d])
    
    # Align to 4h timeframe
    kama_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_slope_1d)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Load 4h data for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate KAMA on 4h
    change_4h = np.abs(np.diff(close_4h, 10))
    volatility_4h = np.sum(np.abs(np.diff(close_4h)), axis=1)
    er_4h = np.divide(change_4h, volatility_4h, out=np.zeros_like(change_4h), where=volatility_4h!=0)
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_4h = (er_4h * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama_4h = np.full_like(close_4h, np.nan)
    kama_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        if not np.isnan(sc_4h[i-1]):
            kama_4h[i] = kama_4h[i-1] + sc_4h[i-1] * (close_4h[i] - kama_4h[i-1])
        else:
            kama_4h[i] = kama_4h[i-1]
    
    # Align 4h KAMA to lower timeframe (if needed, but we're using 4h as primary)
    # Since we're using 4h as primary timeframe, we need to align it to the actual prices timeframe
    # But prices is already at 4h? Let's check the timeframe - we'll set timeframe="4h" below
    # For now, we'll assume prices is at 4h and use kama_4h directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama_slope_1d_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            i >= len(kama_4h) or np.isnan(kama_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # 1d KAMA slope for trend direction
        upward_trend = kama_slope_1d_aligned[i] > 0
        downward_trend = kama_slope_1d_aligned[i] < 0
        
        if position == 0:
            # Long conditions: upward trend on 1d, price > KAMA on 4h, volume confirmation
            if upward_trend and price > kama_4h[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: downward trend on 1d, price < KAMA on 4h, volume confirmation
            elif downward_trend and price < kama_4h[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below KAMA on 4h
            if price < kama_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above KAMA on 4h
            if price > kama_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KAMA_Direction_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0