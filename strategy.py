#!/usr/bin/env python3
# 6h_supertrend_donchian_volume_v1
# Hypothesis: 6h Supertrend(ATR=10, mult=3.0) for trend direction + 6h Donchian(20) breakout for entry timing + volume confirmation.
# Uses 6h timeframe to balance responsiveness and noise reduction. Supertrend provides clear trend state with built-in ATR-based stops,
# Donchian breakouts capture momentum in direction of trend, volume spike filters weak breakouts. Designed for 12-37 trades/year.
# Works in bull/bear markets: Supertrend adapts to volatility, Donchian breakouts work in both directions, volume confirmation avoids fakeouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
            
    return supertrend, direction, atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for indicator calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:  # Need enough for Supertrend(10) and Donchian(20)
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 6h Supertrend(10, 3.0)
    supertrend_6h, direction_6h, atr_6h = calculate_supertrend(high_6h, low_6h, close_6h, 10, 3.0)
    
    # Align 6h Supertrend and direction to 6h timeframe (completed 6h candle only)
    supertrend_6h_aligned = align_htf_to_ltf(prices, df_6h, supertrend_6h)
    direction_6h_aligned = align_htf_to_ltf(prices, df_6h, direction_6h)
    
    # Calculate 6h Donchian(20) channels
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Volume spike detection (20-period volume average on 6h)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (vol_ma_20 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_6h, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_6h_aligned[i]) or np.isnan(direction_6h_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Supertrend (trend reversal)
            if close[i] < supertrend_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Supertrend (trend reversal)
            if close[i] > supertrend_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high, Supertrend uptrend, with volume spike
            if (close[i] > donchian_high_aligned[i]) and (direction_6h_aligned[i] == 1) and vol_spike_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low, Supertrend downtrend, with volume spike
            elif (close[i] < donchian_low_aligned[i]) and (direction_6h_aligned[i] == -1) and vol_spike_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals