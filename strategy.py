#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V2
Hypothesis: 4h Camarilla R1/S1 breakouts with volume confirmation (>1.4x 20-period volume MA) and choppiness regime filter (CHOP(14) between 38.2 and 61.8 for ranging markets). 
ATR-based stoploss (exit when price moves 2.0*ATR against position). 
Camarilla pivots from 1d HTF provide institutional support/resistance levels. 
Chop filter ensures we only trade in ranging markets where mean reversion works. 
Volume confirmation reduces false breakouts. Target 20-50 trades/year (80-200 total over 4 years).
Uses 4h primary timeframe with 1d HTF for Camarilla calculation and chop regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla and chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rng_1d = high_1d - low_1d
    camarilla_r1 = close_1d + (rng_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (rng_1d * 1.1 / 12)
    
    # Align Camarilla levels (no extra delay needed for pivot points)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    # We'll compute a simplified version: CHOP = 100 * log10(ATR_sum / (HH - LL)) / log10(14)
    # Where ATR_sum = sum of true range over 14 periods
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    # Prepend first TR as high-low for simplicity
    tr1 = np.concatenate([[high_1d[0] - low_1d[0]], tr1])
    
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = hh_14 - ll_14
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    chop = 100 * np.log10(atr_sum / denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14) for stoploss
    tr4h = np.maximum(high_4h[1:] - low_4h[1:], np.abs(high_4h[1:] - close_4h[:-1]))
    tr4h = np.maximum(tr4h, np.abs(low_4h[1:] - close_4h[:-1]))
    tr4h = np.concatenate([[high_4h[0] - low_4h[0]], tr4h])
    atr_14 = pd.Series(tr4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.4 * vol_ma[i]  # volume confirmation
        
        # Choppiness regime: only trade when market is ranging (38.2 <= CHOP <= 61.8)
        chop_ok = (chop_aligned[i] >= 38.2) and (chop_aligned[i] <= 61.8)
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume confirmation + chop regime
            if price > camarilla_r1_aligned[i] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S1 + volume confirmation + chop regime
            elif price < camarilla_s1_aligned[i] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # ATR-based stoploss: exit if price drops 2.0*ATR below entry
            if price < entry_price - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            # Exit long: price breaks below Camarilla S1 (mean reversion target)
            elif price < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # ATR-based stoploss: exit if price rises 2.0*ATR above entry
            if price > entry_price + 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            # Exit short: price breaks above Camarilla R1 (mean reversion target)
            elif price > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V2"
timeframe = "4h"
leverage = 1.0