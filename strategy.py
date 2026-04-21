#!/usr/bin/env python3
"""
12h_KAMA_Trend_Chop_Regime_v1
Hypothesis: 12h KAMA trend direction filtered by 1d choppiness regime (CHOP > 61.8 = range, < 38.2 = trend).
In trending regime (CHOP < 38.2), follow KAMA direction. In ranging regime (CHOP > 61.8), mean-revert at Bollinger Bands (20,2).
Uses ATR-based stoploss (2.0x) and discrete position sizing (0.25) to minimize fee drag.
Designed to work in both bull and bear markets via adaptive regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime and Bollinger Bands)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # === 1d OHLC for chop regime and Bollinger Bands ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h KAMA for trend direction ===
    close = prices['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align 12h KAMA to itself (no alignment needed as it's already on 12h)
    kama_aligned = kama  # Already on 12h timeframe
    
    # === 1d Choppiness Index (CHOP) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Pad first value
    tr = np.concatenate([[np.nan], tr[:-1]])
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where((hh_14 - ll_14) != 0, 
                    100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14), 
                    50)
    
    # Align 1d chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 1d Bollinger Bands (20,2) for mean reversion in ranging markets ===
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr[:-1]])  # First TR is NaN
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Regime-based logic
            if chop_aligned[i] < 38.2:  # Trending regime
                # Follow KAMA direction
                if price > kama_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < kama_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif chop_aligned[i] > 61.8:  # Ranging regime
                # Mean revert at Bollinger Bands
                if price <= lower_bb_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price >= upper_bb_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions based on regime
            elif chop_aligned[i] < 38.2:  # Trending regime - exit when price crosses below KAMA
                if price < kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop_aligned[i] > 61.8:  # Ranging regime - exit when price reaches middle BB
                middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
                if price >= middle_bb:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions based on regime
            elif chop_aligned[i] < 38.2:  # Trending regime - exit when price crosses above KAMA
                if price > kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop_aligned[i] > 61.8:  # Ranging regime - exit when price reaches middle BB
                middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
                if price <= middle_bb:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_Chop_Regime_v1"
timeframe = "12h"
leverage = 1.0