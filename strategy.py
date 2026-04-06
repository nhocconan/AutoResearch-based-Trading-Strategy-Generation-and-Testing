#!/usr/bin/env python3
"""
6h Donchian(20) breakout with weekly pivot direction and volume confirmation
Hypothesis: Donchian breakouts capture institutional momentum, filtered by weekly pivot direction (bullish/bearish bias) and volume confirmation for conviction. Works in bull (buy breakouts above pivot) and bear (sell breakdowns below pivot). Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])):
            pivot_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
            r1_1w[i] = 2 * pivot_1w[i] - low_1w[i]
            s1_1w[i] = 2 * pivot_1w[i] - high_1w[i]
    
    # Weekly bias: above pivot = bullish, below = bearish
    bias_1w = np.where(close_1w > pivot_1w, 1, -1)
    
    # Align weekly bias to 6h timeframe
    bias_1w_aligned = align_htf_to_ltf(prices, df_1w, bias_1w)
    
    # Get weekly volume for confirmation
    volume_1w = df_1w['volume'].values
    
    # 4-week average volume on weekly
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    for i in range(4, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-4:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(bias_1w_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 4-week average weekly volume (scaled)
        # Scale weekly volume to 6h: approx 1/28 of weekly volume (4 weeks * 7 days * 4 bars/day)
        vol_threshold = vol_ma_1w_aligned[i] / 28.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly bias
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                bias_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly bias
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                bias_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 24 bars flat
            if bars_since_entry >= 24:
                # Breakout entries: upper/lower with weekly bias
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish weekly bias + volume
                if bull_breakout and bias_1w_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly bias + volume
                elif bear_breakout and bias_1w_aligned[i] == -1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals