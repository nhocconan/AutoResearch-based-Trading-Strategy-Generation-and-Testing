#!/usr/bin/env python3
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
    
    # Get daily data for pivot points (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe (use previous day's levels)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(volume_ma20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # ATR filter: only trade when volatility is reasonable (not too low)
        atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
        volatility_filter = atr[i] > (0.5 * atr_ma[i]) if not np.isnan(atr_ma[i]) else True
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and volatility filter
            if close[i] > r1_4h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and volatility filter
            elif close[i] < s1_4h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot point
            if close[i] < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot point
            if close[i] > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0