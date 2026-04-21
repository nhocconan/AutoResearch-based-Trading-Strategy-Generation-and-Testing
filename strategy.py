#!/usr/bin/env python3
"""
6h_Liquidity_Zone_Reversal
Hypothesis: Price respects liquidity zones (equal highs/lows) as support/resistance. 
In ranging markets (CHOP > 61.8), price reverses from these zones with volume confirmation.
In trending markets (CHOP < 38.2), breakouts of these zones continue the trend.
Uses 1d CHOP for regime filter and 6h equal highs/lows for liquidity zones.
Designed for low trade frequency (target: 20-40/year) with clear entry/exit rules.
Works in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for CHOP regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period CHOP on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR and Chop calculation
    atr_1d = np.zeros_like(close_1d)
    for i in range(len(tr)):
        if i < 14:
            atr_1d[i] = np.mean(tr[:i+1])
        else:
            atr_1d[i] = np.mean(tr[i-14:i])
    
    # Chop = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_1d = np.full_like(close_1d, 50.0)  # default neutral
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(atr_1d[i-13:i+1])
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high > min_low:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate liquidity zones (equal highs/lows) - look for price levels tested 2+ times
    liquidity_long = np.zeros(n, dtype=bool)   # support from equal lows
    liquidity_short = np.zeros(n, dtype=bool)  # resistance from equal highs
    
    # Scan for equal lows (within 0.1% tolerance) - support
    for i in range(20, n):
        lookback = min(60, i)  # look back max 60 periods (~10 days)
        for j in range(i-lookback, i):
            if j < 0:
                continue
            # Check if current low equals a prior low within tolerance
            if abs(low[i] - low[j]) / low[i] < 0.001:
                liquidity_long[i] = True
                break
    
    # Scan for equal highs (within 0.1% tolerance) - resistance
    for i in range(20, n):
        lookback = min(60, i)
        for j in range(i-lookback, i):
            if j < 0:
                continue
            if abs(high[i] - high[j]) / high[i] < 0.001:
                liquidity_short[i] = True
                break
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):  # Start after lookback period
        # Skip if NaN in critical values
        if np.isnan(chop_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        chop = chop_1d_aligned[i]
        vol_ok = volume_filter[i]
        liq_long = liquidity_long[i]
        liq_short = liquidity_short[i]
        
        if position == 0:
            # Regime-based logic
            if chop > 61.8:  # Ranging market - mean reversion from liquidity zones
                # Long at support (equal lows) with volume
                if liq_long and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short at resistance (equal highs) with volume
                elif liq_short and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif chop < 38.2:  # Trending market - breakout continuation
                # Long on breakout of resistance (equal highs) with volume
                if liq_short and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short on breakdown of support (equal lows) with volume
                elif liq_long and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            # In transition zone (38.2-61.8), stay neutral
        
        elif position == 1:
            # Long exit conditions
            if chop > 61.8:  # In range, exit at opposite liquidity zone
                if liq_short:  # Hit resistance
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Trending or transition, trail with liquidity
                if liq_short and vol_ok:  # Strong resistance
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if chop > 61.8:  # In range, exit at opposite liquidity zone
                if liq_long:  # Hit support
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Trending or transition, trail with liquidity
                if liq_long and vol_ok:  # Strong support
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Liquidity_Zone_Reversal"
timeframe = "6h"
leverage = 1.0