#!/usr/bin/env python3
"""
12h_KAMA_Regime_Chop_VolumeBreakout
Hypothesis: On 12h timeframe, use KAMA trend direction as primary filter, combined with 1d chop regime and volume breakout for entries.
Long when: KAMA(12h) rising, 1d chop < 61.8 (trending regime), and volume > 2.0x 20-period MA.
Short when: KAMA(12h) falling, 1d chop < 61.8 (trending regime), and volume > 2.0x 20-period MA.
Uses ATR-based stop (2.5x) and discrete position sizing (0.25) to minimize fee churn.
Designed for low trade frequency (<30/year) to work in both bull and bear markets via regime alignment.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h KAMA for trend direction ===
    close_12h = get_htf_data(prices, '12h')['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_12h, 1)), axis=1)  # 10-period sum of abs changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # Start after 10 periods
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), kama)
    
    # === 1d Choppiness Index (CHOP) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop calculation
    chop = np.full_like(close_1d, np.nan)
    mask = (hh - ll) != 0
    chop[mask] = 100 * np.log10(tr_sum[mask] / (hh[mask] - ll[mask])) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h ATR (20-period) for stoploss ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === Volume confirmation (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(kama_12h_aligned[i-1]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = prices['close'].iloc[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        kama_val = kama_12h_aligned[i]
        kama_prev = kama_12h_aligned[i-1]
        chop_val = chop_aligned[i]
        atr_val = atr_12h_aligned[i]
        
        # KAMA trend direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        # Chop regime: trending if CHOP < 61.8
        chop_trending = chop_val < 61.8
        # Volume breakout: current volume > 2.0x average
        volume_breakout = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: KAMA rising, trending regime, volume breakout
            long_condition = kama_rising and chop_trending and volume_breakout
            # Short: KAMA falling, trending regime, volume breakout
            short_condition = kama_falling and chop_trending and volume_breakout
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (KAMA falling)
                elif kama_falling:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (KAMA rising)
                elif kama_rising:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Regime_Chop_VolumeBreakout"
timeframe = "12h"
leverage = 1.0