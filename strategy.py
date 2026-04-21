#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_ChopFilter_V1
Hypothesis: Camarilla R1/S1 breakout with volume confirmation (>1.5x 20-bar MA) and chop regime filter (CHOP(14) > 61.8) works on 4h timeframe for BTC and ETH in both bull and bear markets. Uses 12h timeframe for chop calculation to avoid look-ahead. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data once for chop regime (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate typical price for Camarilla (using 12h as proxy for 1d - we need actual 1d)
    # Since we don't have 1d directly, we'll use 12h to approximate but align properly
    # Actually, we need 1d data for proper Camarilla - let's load it
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
        
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * range_1d / 12
    camarilla_s1 = close_1d - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate chop regime on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for chop calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Chop = 100 * log10(sum(TR14) / (max_high14 - min_low14)) / log10(14)
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(atr_14) - np.log10(max_high_14 - min_low_14)) / np.log10(14)
    chop[np.isnan(chop) | np.isinf(chop)] = 50  # neutral when invalid
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Chop regime: range-bound market (CHOP > 61.8)
        chop_ok = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume in choppy market
            if price > r1_aligned[i]:
                if chop_ok and volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S1 with volume in choppy market
            elif price < s1_aligned[i]:
                if chop_ok and volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price moves back below R1 or stoploss
            if price < r1_aligned[i] or price < prices['close'].iloc[i-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price moves back above S1 or stoploss
            if price > s1_aligned[i] or price > prices['close'].iloc[i-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0