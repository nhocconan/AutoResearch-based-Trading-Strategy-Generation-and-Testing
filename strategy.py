#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_Regime
Hypothesis: Combines Camarilla pivot levels from 1-day with volume confirmation and Choppiness regime filter.
In trending markets (CHOP < 38.2), breaks of H4/L4 levels with volume > 1.5x average trigger entries.
In ranging markets (CHOP > 61.8), reversals at H3/L3 levels with volume confirmation trigger mean-reversion trades.
Uses 4h timeframe for signals, 1d for pivots and regime. Target: 20-40 trades/year (80-160 total over 4 years).
Works in both bull and bear markets by adapting to regime.
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
    
    # Get daily data for Camarilla pivots and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # H2 = Close + 1.083 * (High - Low)
    # L2 = Close - 1.083 * (High - Low)
    # H1 = Close + 1.0416 * (High - Low)
    # L1 = Close - 1.0416 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    
    range_1d = high_1d - low_1d
    H4 = close_1d + 1.5 * range_1d
    L4 = close_1d - 1.5 * range_1d
    H3 = close_1d + 1.25 * range_1d
    L3 = close_1d - 1.25 * range_1d
    H2 = close_1d + 1.083 * range_1d
    L2 = close_1d - 1.083 * range_1d
    H1 = close_1d + 1.0416 * range_1d
    L1 = close_1d - 1.0416 * range_1d
    Pivot = (high_1d + low_1d + close_1d) / 3
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR1_sum / (14 * ATR14)) / log10(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[0], tr1])  # first period TR = range
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum()
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean()
    chop = 100 * np.log10(atr1 / (14 * atr14)) / np.log10(14)
    chop = chop.values
    
    # Regime: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    trending = chop < 38.2
    ranging = chop > 61.8
    
    # Get 4h data for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume_4h > (vol_ma_20 * 1.5)
    
    # Align all signals to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending)
    ranging_aligned = align_htf_to_ltf(prices, df_1d, ranging)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_4h, volume_expansion)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or \
           np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or \
           np.isnan(trending_aligned[i]) or np.isnan(ranging_aligned[i]) or \
           np.isnan(volume_expansion_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trading logic based on regime
        if trending_aligned[i]:
            # Trending market: breakout of H4/L4 with volume expansion
            if close[i] > H4_aligned[i] and volume_expansion_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif close[i] < L4_aligned[i] and volume_expansion_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            else:
                # Hold position
                signals[i] = position_size * position
        elif ranging_aligned[i]:
            # Ranging market: reversal at H3/L3 with volume expansion
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                # Inside H3-L3 range, look for reversals at boundaries
                if close[i] <= L3_aligned[i] * 1.001 and volume_expansion_aligned[i]:  # Near L3, potential long
                    if position != 1:
                        position = 1
                        signals[i] = position_size
                    else:
                        signals[i] = position_size
                elif close[i] >= H3_aligned[i] * 0.999 and volume_expansion_aligned[i]:  # Near H3, potential short
                    if position != -1:
                        position = -1
                        signals[i] = -position_size
                    else:
                        signals[i] = -position_size
                else:
                    # Hold or flat
                    signals[i] = position_size * position
            else:
                # Outside H3-L3, wait for re-entry
                signals[i] = 0.0
        else:
            # Choppy/transition - no trade
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0