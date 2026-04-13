#!/usr/bin/env python3
"""
4h_1d_Volume_Weighted_Pivot_Breakout
Hypothesis: Trade breakouts from 1d Volume Weighted Average Price (VWAP) on 4h timeframe with volume confirmation and volatility filter.
VWAP acts as dynamic support/resistance where institutional traders operate. Breakouts above VWAP with volume expansion indicate
bullish momentum; breakdowns below VWAP indicate bearish momentum. Volatility filter avoids low-momentum chop. Designed to work
in both bull and bear markets by following momentum. Target: 25-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP for given period."""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    vwap = np.cumsum(vwap_numerator) / np.cumsum(vwap_denominator)
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d)
    vwap_1d_last = vwap_1d[-1] if len(vwap_1d) > 0 else np.nan
    
    # Align VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, vwap_1d_last))
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    # Volatility filter: ATR ratio > 0.8 (avoid low volatility chop)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (atr_ma_50 * 0.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(volume_expansion[i]) or 
            np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long: price above VWAP with volume expansion and sufficient volatility
        long_condition = (close[i] > vwap_1d_aligned[i]) and volume_expansion[i] and volatility_filter[i]
        
        # Short: price below VWAP with volume expansion and sufficient volatility
        short_condition = (close[i] < vwap_1d_aligned[i]) and volume_expansion[i] and volatility_filter[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Volume_Weighted_Pivot_Breakout"
timeframe = "4h"
leverage = 1.0