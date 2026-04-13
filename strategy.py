#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Regime
Hypothesis: Combines daily Camarilla pivot levels with 4h breakout confirmation and choppiness regime filter.
In low volatility regimes (Chop > 61.8), trades mean-reversion off H3/L3 levels.
In trending regimes (Chop < 38.2), trades breakouts of H4/L4 levels.
Uses volume confirmation to avoid false breaks. Works in both bull and bear markets by adapting to regime.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    # H2 = Close + 1.1 * (High - Low) / 6
    # L2 = Close - 1.1 * (High - Low) / 6
    # H1 = Close + 1.1 * (High - Low) / 12
    # L1 = Close - 1.1 * (High - Low) / 12
    
    range_1d = high_1d - low_1d
    H4 = close_1d + 1.1 * range_1d / 2
    L4 = close_1d - 1.1 * range_1d / 2
    H3 = close_1d + 1.1 * range_1d / 4
    L3 = close_1d - 1.1 * range_1d / 4
    H2 = close_1d + 1.1 * range_1d / 6
    L2 = close_1d - 1.1 * range_1d / 6
    H1 = close_1d + 1.1 * range_1d / 12
    L1 = close_1d - 1.1 * range_1d / 12
    
    # Get 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4-period average volume for confirmation
    vol_ma_4_4h = pd.Series(volume_4h).rolling(window=4, min_periods=4).mean()
    volume_confirm = volume_4h > vol_ma_4_4h
    
    # Calculate Choppiness Index on daily (14-period)
    # Chop = 100 * log10(sum(ATR1) / (n * (HH - LL))) / log10(n)
    # Simplified: Chop > 61.8 = ranging, Chop < 38.2 = trending
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    # Pad TR array to match length
    tr_1d = np.concatenate([[0], tr_1d])
    
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean()
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    
    chop = 100 * np.log10(atr_14 * 14 / (hh_14 - ll_14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Regime: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (breakout)
    ranging_market = chop > 61.8
    trending_market = chop < 38.2
    
    # Align all daily data to 4h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging_market)
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending_market)
    
    # Align 4h volume confirmation
    volume_confirm_aligned = align_htf_to_ltf(prices, df_4h, volume_confirm.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or \
           np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or \
           np.isnan(ranging_aligned[i]) or np.isnan(trending_aligned[i]) or \
           np.isnan(volume_confirm_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine regime
        is_ranging = ranging_aligned[i] > 0.5
        is_trending = trending_aligned[i] > 0.5
        
        # Initialize entry signals
        long_entry = False
        short_entry = False
        
        if is_ranging:
            # Ranging market: mean reversion off H3/L3
            # Long when price crosses below L3 and closes back above it with volume
            if close[i] > L3_aligned[i] and close[i-1] <= L3_aligned[i-1] and volume_confirm_aligned[i]:
                long_entry = True
            # Short when price crosses above H3 and closes back below it with volume
            elif close[i] < H3_aligned[i] and close[i-1] >= H3_aligned[i-1] and volume_confirm_aligned[i]:
                short_entry = True
        elif is_trending:
            # Trending market: breakout of H4/L4
            # Long when price breaks above H4 with volume
            if close[i] > H4_aligned[i] and close[i-1] <= H4_aligned[i-1] and volume_confirm_aligned[i]:
                long_entry = True
            # Short when price breaks below L4 with volume
            elif close[i] < L4_aligned[i] and close[i-1] >= L4_aligned[i-1] and volume_confirm_aligned[i]:
                short_entry = True
        
        # Exit conditions: opposite signal or loss of momentum
        exit_long = False
        exit_short = False
        
        if is_ranging:
            # Exit long when price reaches H3
            if close[i] >= H3_aligned[i]:
                exit_long = True
            # Exit short when price reaches L3
            if close[i] <= L3_aligned[i]:
                exit_short = True
        else:
            # In trending, trail with opposite level touch
            # Exit long when price touches L4
            if close[i] <= L4_aligned[i]:
                exit_long = True
            # Exit short when price touches H4
            if close[i] >= H4_aligned[i]:
                exit_short = True
        
        # Update position
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_Regime"
timeframe = "4h"
leverage = 1.0