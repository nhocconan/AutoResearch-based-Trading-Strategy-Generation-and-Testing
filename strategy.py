#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1w Supertrend trend filter and 1d Donchian breakout.
# Long: Price closes above Donchian(20) upper band + Supertrend bullish (1w) + volume > 1.5x average.
# Short: Price closes below Donchian(20) lower band + Supertrend bearish (1w) + volume > 1.5x average.
# Uses weekly Supertrend for trend direction, daily Donchian for breakout signals, volume confirmation.
# Position size: 0.25 to manage drawdown. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Supertrend calculation (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # Calculate True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR
    atr = np.full(len(close_1w), np.nan)
    for i in range(atr_period, len(close_1w)):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate Supertrend
    supertrend = np.full(len(close_1w), np.nan)
    direction = np.full(len(close_1w), np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    for i in range(atr_period, len(close_1w)):
        if np.isnan(atr[i]):
            continue
            
        upperband = (high_1w[i] + low_1w[i]) / 2 + multiplier * atr[i]
        lowerband = (high_1w[i] + low_1w[i]) / 2 - multiplier * atr[i]
        
        if i == atr_period:
            supertrend[i] = upperband
            direction[i] = -1  # start in downtrend
        else:
            if close_1w[i-1] > supertrend[i-1]:
                supertrend[i] = lowerband
                direction[i] = 1
            else:
                supertrend[i] = upperband
                direction[i] = -1
                
            # Adjust bands
            if direction[i] == 1 and supertrend[i] < supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
            if direction[i] == -1 and supertrend[i] > supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
                
            # Update direction based on close
            if close_1w[i] > supertrend[i]:
                direction[i] = 1
            else:
                direction[i] = -1
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period)
    donchian_upper = np.full(len(close_1d), np.nan)
    donchian_lower = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        donchian_upper[i] = np.max(high_1d[i-20:i])
        donchian_lower[i] = np.min(low_1d[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1w Supertrend direction to 12h
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Align 1d Donchian levels to 12h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(direction_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        trend = direction_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price closes above upper band + uptrend + volume confirmation
            if (price > upper and trend == 1 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price closes below lower band + downtrend + volume confirmation
            elif (price < lower and trend == -1 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band (opposite band)
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band (opposite band)
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Supertrend_1d_Donchian_Volume"
timeframe = "12h"
leverage = 1.0