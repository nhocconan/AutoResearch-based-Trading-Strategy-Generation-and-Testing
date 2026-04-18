#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: In both bull and bear markets, breakouts from Donchian channels
with weekly pivot support/resistance and volume confirmation capture strong
momentum moves. The weekly pivot provides institutional-level support/resistance
that holds across market regimes, reducing false breakouts. Volume ensures
institutional participation. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    # R3 = H + 2*(P-L), S3 = L - 2*(H-P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly levels to 6h timeframe (forward-filled from weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        atr_val = atr[i]
        
        # Breakout conditions with volume
        bullish_breakout = (price > donchian_high[i-1]) and vol_ok
        bearish_breakout = (price < donchian_low[i-1]) and vol_ok
        
        # Weekly pivot filter: avoid breakouts against strong S/R
        # For longs: price should be above S1, preferably above pivot
        # For shorts: price should be below R1, preferably below pivot
        long_pivot_ok = price > s1_aligned[i]
        short_pivot_ok = price < r1_aligned[i]
        
        if position == 0:
            # Long: bullish breakout with volume and pivot support
            if bullish_breakout and long_pivot_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout with volume and pivot resistance
            elif bearish_breakout and short_pivot_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to Donchian midpoint or breaks below S1
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if price < midpoint or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to Donchian midpoint or breaks above R1
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if price > midpoint or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0