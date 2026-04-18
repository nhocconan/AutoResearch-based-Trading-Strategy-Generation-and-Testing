#!/usr/bin/env python3
"""
Hypothesis: 12h 144-period SMA trend filter with 1d volume spike and 1d ATR stop.
Long when price > SMA144 + volume spike; short when price < SMA144 + volume spike.
Uses 1d timeframe for trend (more stable than 4h/12h) and volume confirmation.
Designed for 15-25 trades/year to minimize fee drag in ranging 2025 market.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_sma(arr, period):
    """Calculate Simple Moving Average."""
    sma = np.full(len(arr), np.nan)
    if len(arr) < period:
        return sma
    sma[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        sma[i] = sma[i-1] + (arr[i] - arr[i-period]) / period
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 144-period SMA on 1d close
    sma_144_1d = calculate_sma(close_1d, 144)
    
    # Calculate ATR(14) on 1d
    tr1 = np.zeros(len(close_1d))
    tr2 = np.zeros(len(close_1d))
    tr3 = np.zeros(len(close_1d))
    tr1[1:] = np.abs(high_1d[1:] - low_1d[1:])
    tr2[1:] = np.abs(high_1d[1:] - close_1d[:-1])
    tr3[1:] = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average on 1d
    vol_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align to 12h timeframe
    sma_144_1d_12h = align_htf_to_ltf(prices, df_1d, sma_144_1d)
    atr_14_1d_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_20_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 144  # need SMA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_144_1d_12h[i]) or np.isnan(atr_14_1d_12h[i]) or 
            np.isnan(vol_ma_20_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 12h volume > 2.0 * 20-period 1d volume average
        # Note: volume here is 12h volume, vol_ma_20_1d_12h is 1d volume MA aligned to 12h
        vol_spike = volume[i] > 2.0 * vol_ma_20_1d_12h[i]
        
        if position == 0:
            # Long: price above SMA144 + volume spike
            if close[i] > sma_144_1d_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below SMA144 + volume spike
            elif close[i] < sma_144_1d_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below SMA144 or ATR-based stop
            if close[i] <= sma_144_1d_12h[i] or close[i] <= sma_144_1d_12h[i] - 1.5 * atr_14_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above SMA144 or ATR-based stop
            if close[i] >= sma_144_1d_12h[i] or close[i] >= sma_144_1d_12h[i] + 1.5 * atr_14_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_SMA144_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0