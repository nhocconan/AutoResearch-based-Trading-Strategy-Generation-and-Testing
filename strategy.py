#!/usr/bin/env python3
# 4h_Williams_Alligator_1DTrend
# Hypothesis: Williams Alligator (Jaw: SMA13, Teeth: SMA8, Lips: SMA5) crossover signals direction. 
# Long when Lips cross above Teeth AND price above 1-day EMA50 trend; short when Lips cross below Teeth AND price below EMA50.
# Uses volume confirmation (volume > 1.5x 20-period average) to filter false signals.
# Designed for 20-35 trades/year with clear trend and volume to avoid false signals.
# Works in bull via trend continuation and bear via reversals at extremes.

name = "4h_Williams_Alligator_1DTrend"
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
    
    # Load 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1D EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams Alligator components (SMA based)
    # Jaw: SMA13, Teeth: SMA8, Lips: SMA5
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Lips cross above Teeth AND price above 1D EMA50 trend, with volume confirmation
            if lips_val > teeth_val and lips[i-1] <= teeth[i-1] and close[i] > ema50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips cross below Teeth AND price below 1D EMA50 trend, with volume confirmation
            elif lips_val < teeth_val and lips[i-1] >= teeth[i-1] and close[i] < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips cross below Teeth OR price breaks below 1D EMA50
            if lips_val < teeth_val or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips cross above Teeth OR price breaks above 1D EMA50
            if lips_val > teeth_val or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals