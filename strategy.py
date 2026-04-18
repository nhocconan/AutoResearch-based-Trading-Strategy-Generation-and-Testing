#!/usr/bin/env python3
"""
4h_KAMA_TRIX_VolumeSpike_TrendFilter
Long: KAMA rising + TRIX crosses above zero + volume spike
Short: KAMA falling + TRIX crosses below zero + volume spike
Exit: Opposite TRIX cross
Uses KAMA for adaptive trend, TRIX for momentum, volume spike for confirmation.
Designed to capture trend momentum with low trade frequency in both bull and bear markets.
Target: 80-160 total trades over 4 years (20-40/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
    kama = np.full_like(close, np.nan, dtype=float)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_trix(close, length=12):
    """Calculate TRIX indicator"""
    ema1 = pd.Series(close).ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    return trix.values

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA on 4h
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    kama_rising = np.diff(kama, prepend=kama[0]) > 0
    
    # TRIX on 4h
    trix = calculate_trix(close, length=12)
    trix_cross_up = (trix > 0) & (np.roll(trix, 1) <= 0)
    trix_cross_down = (trix < 0) & (np.roll(trix, 1) >= 0)
    
    # Volume spike (2x 20-period average)
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_spike = volume > 2 * vol_ma
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema12_34 = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema12_34_aligned = align_htf_to_ltf(prices, df_12h, ema12_34)
    ema12_34_rising = np.diff(ema12_34_aligned, prepend=ema12_34_aligned[0]) > 0
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema12_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA rising + TRIX crosses up + volume spike + 12h EMA rising
            if kama_rising[i] and trix_cross_up[i] and vol_spike[i] and ema12_34_rising[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + TRIX crosses down + volume spike + 12h EMA falling
            elif not kama_rising[i] and trix_cross_down[i] and vol_spike[i] and not ema12_34_rising[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses down
            if trix_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses up
            if trix_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_TRIX_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0