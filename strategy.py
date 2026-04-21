#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Regime
Hypothesis: Use Camarilla pivot levels (R1/S1) from daily chart with volume confirmation and chop regime filter.
In trending markets (CHOP < 38.2), buy breakouts above R1 or sell breakdowns below S1.
In ranging markets (CHOP > 61.8), fade touches of R1/S1 with mean reversion.
Designed for 4h timeframe to target 20-50 trades/year with strong edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i == 1:
            atr[i] = tr
        else:
            atr[i] = (atr[i-1] * (period-1) + tr) / period
    
    chop = np.full_like(close, 50.0, dtype=float)
    for i in range(period, len(close)):
        atr_sum = np.sum(atr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        if hh - ll != 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    return chop

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    c = close
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for Camarilla levels and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1)
    r1_1d, s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate Choppiness Index for regime filter
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 30-period average
        if i >= 30:
            vol_ma = prices['volume'].iloc[i-30:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        chop = chop_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        if position == 0:
            # Trending market: CHOP < 38.2 - breakout strategy
            if chop < 38.2:
                if price > r1 and volume_ok:
                    signals[i] = 0.25
                    position = 1
                elif price < s1 and volume_ok:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8 - mean reversion at pivot levels
            elif chop > 61.8:
                if price < r1 and price > s1 and volume_ok:
                    # Buy near S1, sell near R1
                    if price <= s1 * 1.002:  # Near S1 with 0.2% buffer
                        signals[i] = 0.25
                        position = 1
                    elif price >= r1 * 0.998:  # Near R1 with 0.2% buffer
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            if chop < 38.2:  # Trending - exit on breakdown below S1
                if price < s1:
                    exit_signal = True
            elif chop > 61.8:  # Ranging - exit when reaching R1 or losing momentum
                if price >= r1 * 0.995:
                    exit_signal = True
            else:  # Transition - exit on opposite pivot touch
                if price >= r1 * 0.995:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            if chop < 38.2:  # Trending - exit on breakout above R1
                if price > r1:
                    exit_signal = True
            elif chop > 61.8:  # Ranging - exit when reaching S1 or losing momentum
                if price <= s1 * 1.005:
                    exit_signal = True
            else:  # Transition - exit on opposite pivot touch
                if price <= s1 * 1.005:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0