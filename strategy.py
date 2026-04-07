#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Spike + Chop Filter
Hypothesis: Breakouts above/below Donchian channels capture momentum with trend filter.
Uses 1d EMA50 as trend filter, volume spikes for confirmation, and chop filter to avoid ranging markets.
Target: 12-37 trades/year per symbol to minimize fee drag.
Works in both bull and bear by following the trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume Spike Detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # 1d EMA50 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Chop Filter (using 1d data)
    df_1d_chop = get_htf_data(prices, '1d')
    high_1d = df_1d_chop['high'].values
    low_1d = df_1d_chop['low'].values
    close_1d = df_1d_chop['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chopping Index (14-period)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_h = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_h - min_l)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d_chop, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Avoid trading in choppy markets (chop > 61.8)
        if chop_aligned[i] > 61.8:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA50
            if close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA50
            if close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high + price above 1d EMA50 + volume spike
            if (close[i] > high_roll[i-1] and 
                close[i] > ema_50_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low + price below 1d EMA50 + volume spike
            elif (close[i] < low_roll[i-1] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals