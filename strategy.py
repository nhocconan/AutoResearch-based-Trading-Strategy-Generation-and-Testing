#!/usr/bin/env python3
"""
6h Donchian(20) breakout with 1-week pivot direction and volume confirmation
Hypothesis: Donchian breakouts capture institutional momentum. Filter by weekly pivot bias (bullish/bearish) and volume for conviction. Weekly pivot provides structural bias from higher timeframe, reducing false breakouts in sideways markets. Works in bull (buy breakouts above weekly pivot resistance) and bear (sell breakdowns below weekly pivot support). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1w_pivot_vol_v2"
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
    
    # Get 1w data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H
    # R2 = P + (H-L), S2 = P - (H-L)
    # R3 = H + 2*(P-L), S3 = L - 2*(H-P)
    pivot = np.full(len(close_1w), np.nan)
    r1 = np.full(len(close_1w), np.nan)
    s1 = np.full(len(close_1w), np.nan)
    r2 = np.full(len(close_1w), np.nan)
    s2 = np.full(len(close_1w), np.nan)
    r3 = np.full(len(close_1w), np.nan)
    s3 = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])):
            pivot[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
            r1[i] = 2 * pivot[i] - low_1w[i]
            s1[i] = 2 * pivot[i] - high_1w[i]
            r2[i] = pivot[i] + (high_1w[i] - low_1w[i])
            s2[i] = pivot[i] - (high_1w[i] - low_1w[i])
            r3[i] = high_1w[i] + 2 * (pivot[i] - low_1w[i])
            s3[i] = low_1w[i] - 2 * (high_1w[i] - pivot[i])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get 1w data for volume confirmation
    volume_1w = df_1w['volume'].values
    
    # 4-week average volume on 1w
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
        if (np.isnan(atr[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1w average volume (scaled)
        # Scale 1w volume to 6h: approx 1/28 of 1w volume (since 28x 6h in 1w)
        vol_threshold = vol_ma_1w_aligned[i] / 28.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly pivot bias
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                close[i] < s1_aligned[i] or  # below weekly S1 = bearish bias
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly pivot bias
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                close[i] > r1_aligned[i] or  # above weekly R1 = bullish bias
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with weekly pivot bias
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with bullish weekly bias (above S1) + volume
                if bull_breakout and close[i] > s1_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly bias (below R1) + volume
                elif bear_breakout and close[i] < r1_aligned[i] and volume_filter:
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
</code>