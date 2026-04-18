#!/usr/bin/env python3
"""
4h_HTF_Target_Zone_Reversal
Hypothesis: Price reverses from multi-day high/low zones (HTF target areas) with volume exhaustion and momentum divergence.
Works in bull markets by catching pullbacks in uptrends and in bear markets by catching bounces in downtrends.
Uses daily high/low as dynamic support/resistance, volume drying up on approach, and RSI divergence for confirmation.
Target: 20-40 trades/year with high win rate via confluence of HTF structure and LTF timing.
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
    
    # Get daily data for HTF reference
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # Start neutral
    
    # Initialize first average
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        if avg_loss[13] != 0:
            rs[13] = avg_gain[13] / avg_loss[13]
            rsi[13] = 100 - (100 / (1 + rs[13]))
    
    # Wilder smoothing
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # Align daily high/low to 4h (these represent prior day's levels)
    # Using already completed daily bars only
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume exhaustion: current volume < 60% of 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_exhaustion = volume < (vol_ma * 0.6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price near daily low, volume exhaustion, RSI oversold
        near_support = (low[i] <= daily_low_aligned[i] * 1.005)  # Within 0.5% of daily low
        rsi_oversold = rsi[i] < 30
        
        if position == 0 and near_support and vol_exhaustion[i] and rsi_oversold:
            signals[i] = 0.25
            position = 1
        
        # Short setup: price near daily high, volume exhaustion, RSI overbought
        elif position == 0:
            near_resistance = (high[i] >= daily_high_aligned[i] * 0.995)  # Within 0.5% of daily high
            rsi_overbought = rsi[i] > 70
            
            if near_resistance and vol_exhaustion[i] and rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        # Exit long: price moves back toward daily high or RSI overbought
        elif position == 1:
            if (high[i] >= daily_high_aligned[i] * 0.995 or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        # Exit short: price moves back toward daily low or RSI oversold
        elif position == -1:
            if (low[i] <= daily_low_aligned[i] * 1.005 or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Target_Zone_Reversal"
timeframe = "4h"
leverage = 1.0