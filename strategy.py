#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Pivot Points for trend direction and 1d ATR breakout for entry.
# Uses weekly pivot levels to determine trend (above/below weekly pivot) and enters on
# 1d ATR breakouts in the direction of the weekly trend. Includes volume confirmation
# to filter breakouts. Designed for low trade frequency (15-25/year) to avoid fee drag.
# Works in bull markets by buying breakouts above weekly pivot and in bear markets
# by selling breakdowns below weekly pivot.
name = "6h_WeeklyPivot_ATRBreakout_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 14-period ATR on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d open for breakout levels
    open_1d = df_1d['open'].values
    open_6h = align_htf_to_ltf(prices, df_1d, open_1d)
    
    # Volume filter: 1.5x 24-period average (1 day of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 14)  # Wait for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(open_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Breakout levels: today's open ± 1.5x ATR
        upper_break = open_6h[i] + 1.5 * atr_6h[i]
        lower_break = open_6h[i] - 1.5 * atr_6h[i]
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above weekly pivot AND breaks above upper level with volume
            if (close[i] > pivot_6h[i] and 
                close[i] > upper_break and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND breaks below lower level with volume
            elif (close[i] < pivot_6h[i] and 
                  close[i] < lower_break and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly pivot or reverses below open
            if close[i] < pivot_6h[i] or close[i] < open_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly pivot or reverses above open
            if close[i] > pivot_6h[i] or close[i] > open_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals