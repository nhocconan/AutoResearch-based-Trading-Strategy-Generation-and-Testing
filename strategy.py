#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Chandelier_Exit_Trend_Volume_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chandelier Exit calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Chandelier Exit on 1d data
    period = 22
    mult = 3.0
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Calculate Chandelier Exit for long and short
    # Long exit: highest high - mult * ATR
    # Short exit: lowest low + mult * ATR
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    
    chandelier_long_exit = highest_high - mult * atr
    chandelier_short_exit = lowest_low + mult * atr
    
    # Calculate trend direction using price relative to Chandelier exits
    # Above long exit = bullish, below short exit = bearish
    trend_bullish = close_1d > chandelier_long_exit
    trend_bearish = close_1d < chandelier_short_exit
    
    # Calculate volume spike indicator (volume > 1.8 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    # Align indicators to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike_aligned[i] > 0.5
        
        if position == 0:
            # Long when bullish trend + volume spike
            if trend_bullish_aligned[i] > 0.5 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when bearish trend + volume spike
            elif trend_bearish_aligned[i] > 0.5 and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when trend turns bearish
            if trend_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when trend turns bullish
            if trend_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals