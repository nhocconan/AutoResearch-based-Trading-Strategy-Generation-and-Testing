#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Crossover_1dTrend_Volume
Hypothesis: Williams Alligator crossover (Jaw/Teeth/Lips) filtered by 1d EMA trend and volume confirmation.
The Alligator uses SMAs with specific periods (13,8,5) and forward shifts (8,5,3) to detect trends.
Works in bull/bear markets by following 1d trend direction. Designed for low trade frequency on 12h timeframe.
Target: 15-25 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "12h_WilliamsAlligator_Crossover_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate ATR(14) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Calculate 1d EMA50 for trend filter (using HTF data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components on price (close)
    # Jaw: SMA(13) shifted 8 bars forward
    # Teeth: SMA(8) shifted 5 bars forward
    # Lips: SMA(5) shifted 3 bars forward
    jaw_raw = np.full(n, np.nan)
    teeth_raw = np.full(n, np.nan)
    lips_raw = np.full(n, np.nan)
    
    for i in range(13, n):
        jaw_raw[i] = np.mean(close[i-13:i])
    for i in range(8, n):
        teeth_raw[i] = np.mean(close[i-8:i])
    for i in range(5, n):
        lips_raw[i] = np.mean(close[i-5:i])
    
    # Apply forward shifts (positive shift means looking into future, so we lag the values)
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    for i in range(8, n):
        if i-8 >= 0 and not np.isnan(jaw_raw[i-8]):
            jaw[i] = jaw_raw[i-8]
    for i in range(5, n):
        if i-5 >= 0 and not np.isnan(teeth_raw[i-5]):
            teeth[i] = teeth_raw[i-5]
    for i in range(3, n):
        if i-3 >= 0 and not np.isnan(lips_raw[i-3]):
            lips[i] = lips_raw[i-3]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 13, 50)  # Ensure volume SMA, Alligator, and EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) with uptrend and volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) with downtrend and volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Lips crosses below Teeth (loss of bullish momentum)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Lips crosses above Teeth (loss of bearish momentum)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals