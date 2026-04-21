#!/usr/bin/env python3
"""
12h_KAMA_Regime_Filter_DonchianExit
Hypothesis: On 12h timeframe, use KAMA trend direction as primary signal, filtered by 1d chop regime (range = mean reversion, trend = trend follow). 
Enter long when KAMA bullish AND chop < 38.2 (trending) OR KAMA bullish AND chop > 61.8 (mean revert to upside); short when KAMA bearish AND chop < 38.2 OR KAMA bearish AND chop > 61.8 (mean revert to downside).
Use 1d Donchian(20) breakout for exit to capture larger moves. Discrete sizing 0.25. Designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for KAMA, chop, Donchian)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d KAMA for trend direction ===
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    er[10:] = change[9:] / np.where(vol[9:] == 0, 1, vol[9:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 1d Choppiness Index (CHOP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # CHOP formula
    chop = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if sum_tr[i] > 0 and hh[i] != ll[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral
    
    # === 1d Donchian(20) for exit ===
    donch_h = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align all 1d indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    donch_h_aligned = align_htf_to_ltf(prices, df_1d, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_1d, donch_l)
    
    # === 12h price for entry/exit ===
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(donch_h_aligned[i]) or np.isnan(donch_l_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        chop_val = chop_aligned[i]
        donch_h_val = donch_h_aligned[i]
        donch_l_val = donch_l_aligned[i]
        
        # Determine market regime and entry conditions
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long conditions: KAMA bullish (price > KAMA)
            if price > kama_val:
                if is_trending or is_ranging:  # enter in both regimes but with different logic
                    signals[i] = 0.25
                    position = 1
            # Short conditions: KAMA bearish (price < KAMA)
            elif price < kama_val:
                if is_trending or is_ranging:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            # Stop and reverse if KAMA turns bearish
            if price < kama_val:
                signals[i] = 0.0
                position = 0
            # Donchian breakout exit (profit target)
            elif price > donch_h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            # Stop and reverse if KAMA turns bullish
            if price > kama_val:
                signals[i] = 0.0
                position = 0
            # Donchian breakout exit (profit target)
            elif price < donch_l_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Regime_Filter_DonchianExit"
timeframe = "12h"
leverage = 1.0