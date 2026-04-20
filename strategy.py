#!/usr/bin/env python3
# 4h_Donchian_RSI_Strategy
# Hypothesis: Donchian channel breakouts with RSI momentum filter capture trends while avoiding false breakouts.
# RSI > 60 for long entries, RSI < 40 for short entries ensures momentum alignment.
# Works in both bull and bear markets by capturing momentum shifts. Low trade frequency minimizes fee drag.

name = "4h_Donchian_RSI_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(high_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to LTF
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate RSI (14-period) on close prices
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i >= 13:  # 14-period average
            avg_gain[i] = np.mean(gain[i-13:i+1])
            avg_loss[i] = np.mean(loss[i-13:i+1])
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure Donchian and RSI are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + RSI > 60 (bullish momentum)
            if close[i] > donchian_high_aligned[i] and rsi[i] > 60:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + RSI < 40 (bearish momentum)
            elif close[i] < donchian_low_aligned[i] and rsi[i] < 40:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or RSI < 50 (momentum fade)
            if close[i] < donchian_low_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or RSI > 50 (momentum fade)
            if close[i] > donchian_high_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals