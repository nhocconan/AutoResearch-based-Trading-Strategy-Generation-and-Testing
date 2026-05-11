#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Use Camarilla R1/S1 levels from daily timeframe for breakout entries, 
confirmed by 1d trend (EMA34) and volume spike. Designed to work in both bull 
and bear markets by capturing breakouts from key support/resistance levels with 
trend alignment and volume confirmation. Targets 20-40 trades/year to minimize 
fee drag while maintaining edge.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    
    # Camarilla levels
    R4 = close + range_val * 1.500
    R3 = close + range_val * 1.250
    R2 = close + range_val * 1.166
    R1 = close + range_val * 1.083
    S1 = close - range_val * 1.083
    S2 = close - range_val * 1.166
    S3 = close - range_val * 1.250
    S4 = close - range_val * 1.500
    
    return R1, S1

def calculate_ema(arr, period):
    """Calculate EMA with proper handling"""
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla R1 and S1
    camarilla_R1 = np.zeros(len(high_1d))
    camarilla_S1 = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        R1, S1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_R1[i] = R1
        camarilla_S1[i] = S1
    
    # Calculate daily EMA34 for trend
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Align daily indicators to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND price above EMA34 AND volume filter
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND price below EMA34 AND volume filter
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR price below EMA34
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 OR price above EMA34
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals