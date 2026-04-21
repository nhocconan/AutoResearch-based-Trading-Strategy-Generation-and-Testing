#!/usr/bin/env python3
"""
4h_HTF_1d_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop
Hypothesis: 4h Camarilla pivot breakouts at R1/S1 with volume spike (>1.5x 20-period volume MA) and chop regime filter (Choppiness Index < 38.2 = trending). 
HTF 1d Camarilla levels provide institutional support/resistance. Volume spikes confirm participation. Chop filter avoids ranging markets. 
ATR-based stoploss via signal=0 when price closes outside ATR bands. Target 20-50 trades/year (80-200 total over 4 years).
Uses 4h primary timeframe with 1d HTF for Camarilla and chop calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots and chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12.0)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d Choppiness Index (14-period) for regime filter ===
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    tr1 = pd.Series(high_1d).rolling(window=14, min_periods=1).max() - pd.Series(low_1d).rolling(window=14, min_periods=1).min()
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop = np.where((atr_14 * 14) > 0, chop, 50.0)  # default to neutral when ATR=0
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_close = np.abs(low_4h - np.roll(close_4h, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr_4h = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(34, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        chop_ok = chop_aligned[i] < 38.2  # trending regime (chop < 38.2)
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + trending regime
            if price > camarilla_r1_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume spike + trending regime
            elif price < camarilla_s1_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: price closes below S1 OR ATR stoploss hit
            if price < camarilla_s1_aligned[i] or price < entry_price - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price closes above R1 OR ATR stoploss hit
            if price > camarilla_r1_aligned[i] or price > entry_price + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop"
timeframe = "4h"
leverage = 1.0