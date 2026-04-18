#!/usr/bin/env python3
"""
4h Price Channel Breakout with Volume Spike and ATR Filter
Hypothesis: Price breaking out of the 20-period high-low channel with volume 
confirmation indicates institutional momentum. ATR filter avoids whipsaws in 
low volatility. Works in both bull and bear markets by following breakout 
direction. Uses 1-day ATR for regime filter to adapt to changing volatility.
"""

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period high/low)
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(20-1, n):
        highest[i] = np.max(high[i-20+1:i+1])
        lowest[i] = np.min(low[i-20+1:i+1])
    
    # ATR for volatility filter (14-period)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Get daily ATR for regime filter (avoid low volatility environments)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR
    atr_1d = np.zeros(len(high_1d))
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volatility regime: only trade when ATR is above its 50-period average
    atr_ma = np.full(n, np.nan)
    for i in range(50-1, n):
        atr_ma[i] = np.mean(atr_1d_aligned[i-50+1:i+1])
    vol_regime = atr_1d_aligned > atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price above 20-period high with volume spike and vol regime
            if (close[i] > highest[i] and 
                vol_spike[i] and 
                vol_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below 20-period low with volume spike and vol regime
            elif (close[i] < lowest[i] and 
                  vol_spike[i] and 
                  vol_regime[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 20-period high
            if close[i] < highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 20-period low
            if close[i] > lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_VolRegime"
timeframe = "4h"
leverage = 1.0