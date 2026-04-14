# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Bull Power (BP) = Close - EMA13; Bear Power (EP) = EMA13 - Low
# Trend filter: 1d ADX > 25 (trending) or ADX < 20 (ranging)
# In trending: enter long when BP > 0 and rising, short when EP > 0 and rising
# In ranging: fade extremes (buy when BP < 0 and turning up, sell when EP < 0 and turning up)
# Volume confirmation: > 1.3x 20-period average
# Designed for 60-120 trades over 4 years with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13  # Close - EMA13
    bear_power = ema13 - low    # EMA13 - Low
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):  # Start after calculations
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Check for NaN values
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned)):
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Momentum: current vs previous power
        bull_momentum = bull_power[i] > bull_power[i-1]
        bear_momentum = bear_power[i] > bear_power[i-1]
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Trending market (ADX > 25): follow momentum
                if adx_1d_aligned > 25:
                    if bull_power[i] > 0 and bull_momentum:
                        position = 1
                        signals[i] = position_size
                    elif bear_power[i] > 0 and bear_momentum:
                        position = -1
                        signals[i] = -position_size
                # Ranging market (ADX < 20): fade extremes
                elif adx_1d_aligned < 20:
                    if bull_power[i] < 0 and bull_momentum:  # BP negative but turning up
                        position = 1
                        signals[i] = position_size
                    elif bear_power[i] < 0 and bear_momentum:  # EP negative but turning up
                        position = -1
                        signals[i] = -position_size
        elif position == 1:  # Long position - exit conditions
            # Exit when power fades or reverses
            if bull_power[i] <= 0 or not bull_momentum:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit conditions
            # Exit when power fades or reverses
            if bear_power[i] <= 0 or not bear_momentum:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dADX_Regime"
timeframe = "6h"
leverage = 1.0