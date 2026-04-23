#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
The Alligator uses SMAs (13,8,5) to identify trend: when Lips > Teeth > Jaw = uptrend, reverse for downtrend.
Combined with 1d EMA50 for higher timeframe trend alignment and volume spike for breakout confirmation.
Designed for 12h timeframe to minimize trades while capturing medium-term trends in both bull and bear markets.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee drag.
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
    
    # Calculate 12h Williams Alligator (SMA 13, 8, 5)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # SMA 13
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values   # SMA 8
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values    # SMA 5
    
    # Align Alligator lines to primary timeframe (previous 12h bar values)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (Alligator bullish) AND price above 1d EMA50 AND volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (Alligator bearish) AND price below 1d EMA50 AND volume spike
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator sleeping (no clear trend) or opposite alignment
            exit_signal = False
            if position == 1:
                # Exit long when Alligator turns bearish or goes sideways
                if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator turns bullish or goes sideways
                if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0