#!/usr/bin/env python3
"""
12h_RsiStDevBreakout
Hypothesis: Combine RSI mean-reversion with standard deviation breakout to capture reversals in both bull and bear markets.
Uses 1d RSI(14) for overbought/oversold conditions and 12h Bollinger Bands(20,2) for breakout confirmation.
Volume filter ensures participation. Designed for low-frequency, high-conviction trades (target: 15-25/year).
Works in bull via oversold bounces and in bear via overbought reversals.
"""

name = "12h_RsiStDevBreakout"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    delta = np.diff(df_1d['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = np.concatenate([[np.nan], rsi_14])  # align with df_1d index
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 12h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Calculate 12h average volume for volume filter (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14), SMA (20), volume avg (20)
    start_idx = max(14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_14_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion condition: RSI extreme
        oversold = rsi_14_aligned[i] < 30
        overbought = rsi_14_aligned[i] > 70
        
        # Breakout condition: price outside Bollinger Bands
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = volume[i] > vol_avg_20[i] * 1.5
        
        if position == 0:
            # Long entry: oversold + downward breakout (mean reversion) + volume
            if oversold and breakout_down and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: overbought + upward breakout (mean reversion) + volume
            elif overbought and breakout_up and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or breakout reverses
            if rsi_14_aligned[i] > 50 or close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or breakout reverses
            if rsi_14_aligned[i] < 50 or close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals