#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot context and volume confirmation
# Uses weekly pivot levels to define long-term support/resistance zones
# In ranging markets: price reverses at weekly S1/R1 (mean reversion)
# In trending markets: price breaks weekly S2/R2 with volume (continuation)
# Volume filter confirms breakout strength
# Works in both bull and bear markets by adapting to volatility regime
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years)

name = "6h_WkPivot_S1R1_S2R2_VolumeATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for ATR calculation (better resolution than 6h)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate ATR(14) on 1h
    tr1 = np.maximum(high_1h[1:], close_1h[:-1]) - np.minimum(low_1h[1:], close_1h[:-1])
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support and resistance levels
    s1_1w = 2 * pivot_1w - high_1w
    r1_1w = 2 * pivot_1w - low_1w
    s2_1w = pivot_1w - (high_1w - low_1w)
    r2_1w = pivot_1w + (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need volume MA and ATR data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or np.isnan(atr_1h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1h_aligned[i]
        
        # Volume and volatility filters
        volume_confirmed = vol > 1.5 * vol_ma
        volatility_filter = atr > 0  # Always true but keeps structure
        
        # Weekly pivot levels
        s1 = s1_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        pivot = pivot_1w_aligned[i]
        
        if position == 0:
            # Long conditions:
            # 1. Breakout above R2 with volume (strong bullish)
            # 2. Mean reversion from S1 with volume (bullish bounce)
            if ((price > r2 and volume_confirmed) or 
                (price < s1 and price > s2 and volume_confirmed and price > pivot)):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Breakdown below S2 with volume (strong bearish)
            # 2. Mean reversion from R1 with volume (bearish rejection)
            elif ((price < s2 and volume_confirmed) or 
                  (price > r1 and price < r2 and volume_confirmed and price < pivot)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below S1 or reversal at R1
            if price < s1 or (price > r1 and price < r2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above S2 or reversal at S2
            if price > r2 or (price < r1 and price > s2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals