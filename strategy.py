#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h Camarilla R1/S1 breakouts with volume confirmation (>1.5x 20-period volume MA) and chop regime filter (Chop > 61.8 = range, mean reversion). 
In ranging markets, fade extreme Camarilla levels (R1/S1) with tight stops. Uses 4h primary timeframe with 12h HTF for chop regime.
Target: 25-40 trades/year (100-160 total over 4 years). Works in both bull (breakouts) and bear/range (mean reversion) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for chop regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Choppy Market Index (Chop) for regime filter ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    range_12h = max_high - min_low
    chop = 100 * np.log10(sum_atr / range_12h) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # === 4h Primary timeframe indicators ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Camarilla levels from previous day (using 12h data approximated)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We'll use 12h bar as proxy for "daily" in crypto 24h market
    prev_close = df_4h['close'].shift(1).values  # previous 4h bar close
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr_4h1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr_4h2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr_4h3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum.reduce([tr_4h1, tr_4h2, tr_4h3])
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            if is_ranging:
                # In ranging market: mean reversion at extreme Camarilla levels
                # Long near S1, short near R1
                if price <= s1[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                elif price >= r1[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            else:
                # In trending market: breakout continuation
                if price > r1[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                elif price < s1[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses mid-point or stoploss hit
            mid_point = (r1[i] + s1[i]) / 2
            if price >= mid_point or price < low_4h[i] - 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses mid-point or stoploss hit
            mid_point = (r1[i] + s1[i]) / 2
            if price <= mid_point or price > high_4h[i] + 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0