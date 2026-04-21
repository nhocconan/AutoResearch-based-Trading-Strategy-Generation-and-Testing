#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop
Hypothesis: Use 4h primary timeframe with 1d Camarilla R1/S1 breakout for momentum capture.
Add volume confirmation (>1.5x 20-bar volume MA) and chop regime filter (CHOP>61.8 = range, CHOP<38.2 = trend).
In ranging markets (CHOP>61.8): mean reversion at R1/S1 levels.
In trending markets (CHOP<38.2): breakout continuation.
Position size 0.25 balances risk/return. Target 20-50 trades/year per symbol.
Uses ATR-based stoploss (2*ATR) via signal=0 when stop hit.
Works in bull/bear via regime adaptation and volume filter reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1, Pivot) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0  # R1 = Close + 1.1*(High-Low)/12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0  # S1 = Close - 1.1*(High-Low)/12
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss and chop calculation
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # first TR is NaN
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Update ATR-based stoploss
        if position == 1:
            atr_stop = entry_price - 2.0 * atr[i]
        elif position == -1:
            atr_stop = entry_price + 2.0 * atr[i]
        
        # Check stoploss hit
        if position == 1 and price < atr_stop:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > atr_stop:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Determine market regime
            is_ranging = chop[i] > 61.8
            is_trending = chop[i] < 38.2
            
            if is_ranging:
                # In ranging markets: mean reversion at R1/S1
                # Long: price crosses above S1 with volume confirmation
                if price > s1_1d_aligned[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price crosses below R1 with volume confirmation
                elif price < r1_1d_aligned[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:
                # In trending markets: breakout continuation
                # Long: price breaks above R1 with volume confirmation
                if price > r1_1d_aligned[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short: price breaks below S1 with volume confirmation
                elif price < s1_1d_aligned[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Exit conditions
            exit_signal = False
            # Mean reversion: price returns to pivot level
            if abs(price - pivot_1d_aligned[i]) < 0.1 * atr[i]:
                exit_signal = True
            # Trend exhaustion: chop increases above 50 (leaving trend)
            elif chop[i] > 50.0:
                exit_signal = True
            # Hold position otherwise
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            exit_signal = False
            # Mean reversion: price returns to pivot level
            if abs(price - pivot_1d_aligned[i]) < 0.1 * atr[i]:
                exit_signal = True
            # Trend exhaustion: chop increases above 50 (leaving trend)
            elif chop[i] > 50.0:
                exit_signal = True
            # Hold position otherwise
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop"
timeframe = "4h"
leverage = 1.0