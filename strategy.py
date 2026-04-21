#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Spike_Regime_ATR_V1
Hypothesis: 12h Camarilla R1/S1 breakouts with volume spike (>1.5x 20-period MA) and chop regime filter (CHOP > 61.8 = ranging market for mean reversion). 
Long when price breaks above R1 in ranging market with volume confirmation. 
Short when price breaks below S1 in ranging market with volume confirmation. 
ATR-based stoploss and profit target. Uses 12h primary timeframe with 1d HTF for Camarilla calculation.
Designed for low-frequency, high-conviction trades (12-37/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot points)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) for regime filter - ranging market > 61.8
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_12h[1:] - close_12h[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # same length as close
    
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values) / np.log10(14) / np.log10(range_14)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)  # default to neutral if invalid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_12h = pd.Series(tr1).rolling(window=20, min_periods=20).mean().values  # ATR for stops
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(chop[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop_ok = chop[i] > 61.8  # ranging market regime (mean reversion favorable)
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + ranging market
            if price > r1_1d_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + ranging market
            elif price < s1_1d_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stoploss (2*ATR) or profit target (3*ATR) or regime change
            stop_loss = entry_price - 2.0 * atr_12h[i]
            take_profit = entry_price + 3.0 * atr_12h[i]
            
            if price <= stop_loss or price >= take_profit or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stoploss (2*ATR) or profit target (3*ATR) or regime change
            stop_loss = entry_price + 2.0 * atr_12h[i]
            take_profit = entry_price - 3.0 * atr_12h[i]
            
            if price >= stop_loss or price <= take_profit or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Spike_Regime_ATR_V1"
timeframe = "12h"
leverage = 1.0