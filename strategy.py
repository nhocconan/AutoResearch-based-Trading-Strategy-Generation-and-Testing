#!/usr/bin/env python3
"""
6h_SuperTrend_WeeklyTrend_Filter
Hypothesis: SuperTrend on 6h with weekly trend filter (price above/below weekly EMA200) and volume confirmation.
Only take long when price > weekly EMA200 and SuperTrend gives long signal; short when price < weekly EMA200 and SuperTrend gives short signal.
Aims to capture strong trends while avoiding counter-trend whipsaws in ranging markets. Designed for low frequency (~20-40 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 trend filter
    ema200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[0:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema200_1w[i] = close_1w[i] * alpha + ema200_1w[i-1] * (1 - alpha)
    
    # Get 6h data for SuperTrend calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate ATR(10) for SuperTrend
    atr_period = 10
    tr = np.maximum(
        high_6h[1:] - low_6h[1:],
        np.maximum(
            np.abs(high_6h[1:] - close_6h[:-1]),
            np.abs(low_6h[1:] - close_6h[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.full(len(close_6h), np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[1:atr_period+1])  # Skip first NaN
        for i in range(atr_period, len(atr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # SuperTrend parameters
    factor = 3.0
    
    # Basic upper and lower bands
    hl2 = (high_6h + low_6h) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    # Initialize SuperTrend
    supertrend = np.full(len(close_6h), np.nan)
    direction = np.full(len(close_6h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    if len(close_6h) >= atr_period:
        supertrend[atr_period-1] = hl2[atr_period-1]
        direction[atr_period-1] = 1 if close_6h[atr_period-1] > supertrend[atr_period-1] else -1
        
        for i in range(atr_period, len(close_6h)):
            if close_6h[i-1] > supertrend[i-1]:
                # Previous trend was up
                supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                # Previous trend was down
                supertrend[i] = min(upper_band[i], supertrend[i-1])
            
            # Determine direction
            direction[i] = 1 if close_6h[i] > supertrend[i] else -1
    
    # Align weekly EMA200 and SuperTrend components to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    supertrend_6h_aligned = align_htf_to_ltf(prices, df_6h, supertrend)
    direction_6h_aligned = align_htf_to_ltf(prices, df_6h, direction)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(supertrend_6h_aligned[i]) or 
            np.isnan(direction_6h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: SuperTrend uptrend, price above weekly EMA200, volume spike
            if (direction_6h_aligned[i] == 1 and 
                close[i] > ema200_1w_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: SuperTrend downtrend, price below weekly EMA200, volume spike
            elif (direction_6h_aligned[i] == -1 and 
                  close[i] < ema200_1w_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: SuperTrend turns down OR price crosses below weekly EMA200
            if (direction_6h_aligned[i] == -1 or close[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: SuperTrend turns up OR price crosses above weekly EMA200
            if (direction_6h_aligned[i] == 1 or close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_SuperTrend_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0