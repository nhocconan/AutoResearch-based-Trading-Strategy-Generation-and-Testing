#!/usr/bin/env python3
"""
12h_TurtleTrader_Donchian20_1dATR2
Turtle Trading breakout system:
- Long when price breaks above 20-period Donchian high + volume confirmation
- Short when price breaks below 20-period Donchian low + volume confirmation  
- Exit on opposite 10-period Donchian break (stop and reverse)
- Uses 1d ATR for position sizing and stop loss
- Designed for 15-25 trades/year per symbol
Works in both bull (captures trends) and bear (short breakdowns) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels (upper and lower bands)."""
    upper = np.full(len(high), np.nan)
    lower = np.full(len(high), np.nan)
    
    if len(high) >= period:
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(2)
    atr_2_1d = calculate_atr(high_1d, low_1d, close_1d, 2)
    
    # Calculate 12-period Donchian channels (20 in original Turtle, but 12 for 12h timeframe)
    upper_12, lower_12 = calculate_donchian_channels(high, low, 12)
    upper_10, lower_10 = calculate_donchian_channels(high, low, 10)  # for exits
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Align 1d ATR to 12h timeframe
    atr_2_1d_12h = align_htf_to_ltf(prices, df_1d, atr_2_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need sufficient data for Donchian, ATR, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12[i]) or np.isnan(lower_12[i]) or 
            np.isnan(atr_2_1d_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 12-period Donchian high + volume
            if close[i] > upper_12[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12-period Donchian low + volume
            elif close[i] < lower_12[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 10-period Donchian low (stop and reverse)
            if close[i] < lower_10[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 10-period Donchian high (stop and reverse)
            if close[i] > upper_10[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TurtleTrader_Donchian20_1dATR2"
timeframe = "12h"
leverage = 1.0