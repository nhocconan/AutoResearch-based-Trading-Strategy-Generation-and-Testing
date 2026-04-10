#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# - Donchian(20) from 6h provides clear breakout structure
# - Weekly pivot (from 1w) determines bias: long above weekly pivot, short below
# - 6h ATR-volume (>1.5x 20-period average) confirms institutional participation
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Weekly pivot adds structural bias that works in both bull and bear markets

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w OHLC for weekly pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = pivot_1w + 3 * (high_1w - low_1w)
    s4_1w = pivot_1w - 3 * (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Pre-compute 6h ATR volume for volume confirmation
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Calculate 6h True Range
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h ATR volume: volume / ATR (normalizes volume by volatility)
    atr_volume_6h = volume_6h / atr_6h
    atr_volume_ma_20_6h = pd.Series(atr_volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to LTF
    atr_volume_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_volume_ma_20_6h)
    
    # Pre-compute 6h Donchian channels (20-period)
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(atr_volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h ATR volume for filter (aligned)
        atr_volume_6h_current = atr_volume_6h
        atr_volume_6h_aligned = align_htf_to_ltf(prices, df_1w, atr_volume_6h_current)
        
        # Volume confirmation: current 6h ATR volume > 1.5x 20-period average
        volume_confirm = atr_volume_6h_aligned[i] > 1.5 * atr_volume_ma_aligned[i]
        
        close_price = close_6h[i]
        
        # Determine weekly pivot bias
        above_pivot = close_price > pivot_aligned[i]
        below_pivot = close_price < pivot_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close_price > donchian_upper[i]
        short_breakout = close_price < donchian_lower[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Long breakout AND above weekly pivot AND volume confirmation
            if long_breakout and above_pivot and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: Short breakout AND below weekly pivot AND volume confirmation
            elif short_breakout and below_pivot and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: 
            # 1. Price crosses weekly pivot in opposite direction
            # 2. Donchian breakout in opposite direction
            exit_long = position == 1 and (close_price < pivot_aligned[i] or short_breakout)
            exit_short = position == -1 and (close_price > pivot_aligned[i] or long_breakout)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals