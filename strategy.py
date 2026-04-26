#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
Uses Camarilla pivot levels from 6h timeframe combined with 12h EMA50 trend filter to avoid counter-trend trades.
Volume spike confirms institutional interest. Designed for 50-150 total trades over 4 years (12-37/year) with 
discrete position sizing (0.0, ±0.25). Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Calculate Camarilla levels from previous 6h bar (need high/low/close of completed 6h bar)
        # We'll use the current 6h bar's high/low/close for Camarilla calculation of NEXT bar
        # But to avoid look-ahead, we calculate Camarilla for the completed bar at i-1
        if i >= 1:
            # Get the completed 6h bar at index i-1
            # We need to aggregate 6h data from 5m data, but since we're on 6h timeframe,
            # the prices DataFrame is already 6h bars
            camarilla_range = (high[i-1] - low[i-1]) * 1.1 / 12
            camarilla_R3 = close[i-1] + camarilla_range * 3
            camarilla_S3 = close[i-1] - camarilla_range * 3
        else:
            camarilla_R3 = np.nan
            camarilla_S3 = np.nan
        
        # Long logic: Close breaks above Camarilla R3 + price > 12h EMA50 (uptrend) + volume spike
        if close[i] > camarilla_R3 and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Camarilla S3 + price < 12h EMA50 (downtrend) + volume spike
        elif close[i] < camarilla_S3 and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses 12h EMA50 in opposite direction
        elif position == 1 and close[i] < ema_50_12h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_50_12h_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0