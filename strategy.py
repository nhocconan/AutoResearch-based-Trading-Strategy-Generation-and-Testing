#!/usr/bin/env python3
"""
4h_DeMarker_Trend_Reversal
Hypothesis: Uses DeMarker indicator on 1h timeframe to detect overbought/oversold conditions, 
combined with 4h trend (EMA50) and volume confirmation. Designed for mean-reversion entries 
with trend filtering to work in both bull and bear markets. Targets 20-40 trades/year to 
minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate DeMarker on 1h timeframe
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # DeMarker components
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    prev_close_1h = df_1h['close'].shift(1).values
    
    # DeMax and DeMin
    demax = np.where(high_1h > prev_close_1h, high_1h - prev_close_1h, 0.0)
    demin = np.where(low_1h < prev_close_1h, prev_close_1h - low_1h, 0.0)
    
    # Smoothed DeMax and DeMin (13-period EMA)
    demax_smooth = pd.Series(demax).ewm(span=13, adjust=False, min_periods=13).mean().values
    demin_smooth = pd.Series(demin).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # DeMarker value
    demark = np.where((demax_smooth + demin_smooth) != 0, 
                      demax_smooth / (demax_smooth + demin_smooth), 
                      0.5)
    
    # Align DeMarker to 4h timeframe
    demark_aligned = align_htf_to_ltf(prices, df_1h, demark)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(demark_aligned[i]) or np.isnan(ema50_4h[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        demark_val = demark_aligned[i]
        ema50_val = ema50_4h[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: DeMarker oversold (<0.3), above EMA50 trend, volume confirmation
            if demark_val < 0.3 and close[i] > ema50_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: DeMarker overbought (>0.7), below EMA50 trend, volume confirmation
            elif demark_val > 0.7 and close[i] < ema50_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: DeMarker overbought (>0.7) or below EMA50
            if demark_val > 0.7 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: DeMarker oversold (<0.3) or above EMA50
            if demark_val < 0.3 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DeMarker_Trend_Reversal"
timeframe = "4h"
leverage = 1.0