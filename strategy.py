#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Regime_ATRStop_v2
Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation (>1.5x 20-period volume MA) and 1d chop regime filter (CHOP > 61.8 = range → mean revert at S1/R1). 
In ranging markets (CHOP > 61.8): fade extremes → short R1 breakdown, long S1 bounce. 
In trending markets (CHOP ≤ 61.8): follow breakout → long R1 break, short S1 breakdown.
ATR trailing stop (2.5x ATR) manages risk. Position size 0.25 balances risk/return.
Target ~25-50 trades/year per symbol (100-200 total over 4 years).
Uses 4h primary timeframe with 1d HTF for regime filter (more stable than 12h).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Chop regime (trend/range filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Chop calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla pivot points (R1, S1) from previous day
    # Using previous 4h bar's high, low, close (shifted by 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        chop_val = chop_aligned[i]
        is_ranging = chop_val > 61.8  # CHOP > 61.8 = ranging market
        
        if position == 0:
            if is_ranging:
                # In ranging market: fade extremes
                # Long: price bounces off S1 + volume confirmation
                if price < s1[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                # Short: price rejects at R1 + volume confirmation
                elif price > r1[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
            else:
                # In trending market: follow breakout
                # Long: price breaks above R1 + volume confirmation
                if price > r1[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                # Short: price breaks below S1 + volume confirmation
                elif price < s1[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Regime_ATRStop_v2"
timeframe = "4h"
leverage = 1.0