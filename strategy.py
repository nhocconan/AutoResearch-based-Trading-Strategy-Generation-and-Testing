#!/usr/bin/env python3
"""
6h_Fisher_MeanReversion_With_Volume
Hypothesis: Ehlers Fisher Transform on 12h price identifies extreme momentum states.
During ranging markets (identified by Choppiness Index), Fisher extremes (>|1.5|) 
signal mean reversion opportunities. Volume confirmation filters false signals.
Designed for low trade frequency (10-20/year) to work in both bull and bear 
markets by combining momentum extremes with mean reversion in ranging conditions.
"""

name = "6h_Fisher_MeanReversion_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def fisher_transform(price, period=10):
    """Calculate Ehlers Fisher Transform"""
    # Normalize price to 0-1 range over period
    min_low = pd.Series(price).rolling(window=period, min_periods=period).min()
    max_high = pd.Series(price).rolling(window=period, min_periods=period).max()
    range_val = max_high - min_low
    
    # Avoid division by zero
    range_val = range_val.replace(0, 1e-10)
    
    # Normalize
    value = 2 * ((price - min_low) / range_val - 0.5)
    # Clamp to avoid domain errors in log
    value = np.clip(value, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    # Smooth
    fisher = pd.Series(fisher).ewm(alpha=0.5, adjust=False).mean().values
    return fisher

def choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index to identify ranging vs trending markets"""
    # True Range
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR
    atr = tr.rolling(window=period, min_periods=period).sum()
    
    # Maximum range over period
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    range_max = max_high - min_low
    
    # Choppiness
    chop = 100 * np.log10(atr / range_max) / np.log10(period)
    chop = chop.fillna(50).values  # Fill NaN with neutral value
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Fisher Transform and Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Fisher Transform on 12h close prices
    fisher_raw = fisher_transform(df_12h['close'].values, period=10)
    fisher_12h = align_htf_to_ltf(prices, df_12h, fisher_raw)
    
    # Calculate Choppiness Index on 12h data
    chop_raw = choppiness_index(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h = align_htf_to_ltf(prices, df_12h, chop_raw)
    
    # Volume confirmation: > 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        chop = chop_12h[i]
        fisher = fisher_12h[i]
        
        # Define ranging market: Choppiness > 61.8
        is_ranging = chop > 61.8
        
        if position == 0:
            # LONG: Fisher below -1.5 (oversold) in ranging market with volume
            if is_ranging and fisher < -1.5 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher above 1.5 (overbought) in ranging market with volume
            elif is_ranging and fisher > 1.5 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses above -0.5 (return to neutral) or extreme reversal
            if fisher > -0.5 or (fisher > 1.5 and is_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses below 0.5 (return to neutral) or extreme reversal
            if fisher < 0.5 or (fisher < -1.5 and is_ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals