#!/usr/bin/env python3
"""
1d_KAMA_Direction_Regime_Filter_DonchianExit
Hypothesis: Daily KAMA trend direction filtered by 1-week choppiness regime, with Donchian(20) exits.
KAMA adapts to market noise, reducing whipsaw in choppy conditions. Choppiness filter avoids trend-following in ranging markets.
Donchian breakouts provide clear entry signals aligned with the adaptive trend. Designed for BTC/ETH in both bull and bear markets.
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1-week for choppiness regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Daily KAMA for adaptive trend ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.concatenate([np.full(10, np.nan), er])  # align with original close
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1-week Choppiness regime filter ===
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # True Range
    tr1 = df_1w_high - df_1w_low
    tr2 = np.abs(df_1w_high - np.roll(df_1w_close, 1))
    tr3 = np.abs(df_1w_low - np.roll(df_1w_close, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]  # first period
    
    # Sum of TR over 14 periods
    sum_tr_14 = np.convolve(tr_1w, np.ones(14), 'valid')
    sum_tr_14 = np.concatenate([np.full(13, np.nan), sum_tr_14])
    
    # Highest high and lowest low over 14 periods
    max_hh_14 = np.convolve(df_1w_high, np.ones(14), 'valid')
    max_hh_14 = np.concatenate([np.full(13, np.nan), max_hh_14])
    min_ll_14 = np.convolve(df_1w_low, np.ones(14), 'valid')
    min_ll_14 = np.concatenate([np.full(13, np.nan), min_ll_14])
    range_14 = max_hh_14 - min_ll_14
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / range_14) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    chop = np.where(range_14 == 0, 100, chop)  # avoid division by zero
    
    # Align 1w chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === Daily Donchian(20) for breakout signals ===
    highest_high_20 = np.convolve(close, np.ones(20), 'valid')
    highest_high_20 = np.concatenate([np.full(19, np.nan), highest_high_20])
    lowest_low_20 = np.convolve(close, np.ones(20), 'valid')
    lowest_low_20 = np.concatenate([np.full(19, np.nan), lowest_low_20])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # warmup for KAMA and Donchian
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        chop_val = chop_aligned[i]
        upper_donch = highest_high_20[i]
        lower_donch = lowest_low_20[i]
        
        # Regime filter: only trend-follow when chop < 50 (trending market)
        is_trending = chop_val < 50
        
        if position == 0:
            # Enter long: price above Donchian upper AND above KAMA AND trending regime
            if price > upper_donch and price > kama_val and is_trending:
                signals[i] = 0.25
                position = 1
            # Enter short: price below Donchian lower AND below KAMA AND trending regime
            elif price < lower_donch and price < kama_val and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below Donchian lower OR below KAMA
            if price < lower_donch or price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above Donchian upper OR above KAMA
            if price > upper_donch or price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0