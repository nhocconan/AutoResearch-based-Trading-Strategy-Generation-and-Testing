#!/usr/bin/env python3
"""
1d_KAMA_Regime_Volume_ATRStop
Hypothesis: Daily strategy using Kaufman Adaptive Moving Average (KAMA) trend direction
combined with volume confirmation (>1.8x 20-day average) and choppiness regime filter
(CHOP > 61.8 = ranging market for mean reversion entries). Enters long when price
pulls back to KAMA in uptrend with high volume in ranging market, short when price
rallies to KAMA in downtrend with high volume in ranging market. Uses ATR(14) stoploss
(2.5*ATR) and time-based exit (10 days max hold). Designed for low trade frequency
(15-25/year) to minimize fee drag while capturing mean reversion in ranging markets
and trend continuation in trending markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for regime, 1d for KAMA/volume/ATR)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1w Choppiness Index for regime filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    range_1w = hh_1w - ll_1w
    chop_1w = np.where(
        range_1w > 0,
        100 * np.log10(atr_1w / range_1w) / np.log10(14),
        50  # neutral when range is zero
    )
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === 1d KAMA ( Kaufman Adaptive Moving Average ) ===
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.where(
        volatility > 0,
        change / volatility,
        0  # no volatility = no trend
    )
    # Smoothing constants
    sc = (er * (2.0 / (2 + 1) - 2.0 / (30 + 1)) + 2.0 / (30 + 1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed value
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 1d Indicators (primary timeframe) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(chop_1w_aligned[i]) or np.isnan(kama_aligned[i]) 
            or np.isnan(volume_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close_1d[i]
        
        if position != 0:
            bars_since_entry += 1
        
        if position == 0:
            # Regime filter: only trade in ranging markets (CHOP > 61.8)
            if chop_1w_aligned[i] > 61.8:
                # Long: price pulls back to KAMA from below with volume spike
                long_condition = (
                    price <= kama_aligned[i] * 1.005 and  # within 0.5% above KAMA
                    price > kama_aligned[i] * 0.995 and   # within 0.5% below KAMA
                    close_1d[i-1] < kama_aligned[i-1] and # previous close below KAMA
                    volume_1d[i] > volume_threshold[i]
                )
                # Short: price rallies to KAMA from above with volume spike
                short_condition = (
                    price >= kama_aligned[i] * 0.995 and  # within 0.5% below KAMA
                    price < kama_aligned[i] * 1.005 and   # within 0.5% above KAMA
                    close_1d[i-1] > kama_aligned[i-1] and # previous close above KAMA
                    volume_1d[i] > volume_threshold[i]
                )
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position == 1:
            # Stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            # Time-based exit (max 10 days hold)
            elif bars_since_entry >= 10:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            # Mean reversion exit: price reaches opposite side of KAMA
            elif price >= kama_aligned[i] * 1.01:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            # Time-based exit (max 10 days hold)
            elif bars_since_entry >= 10:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            # Mean reversion exit: price reaches opposite side of KAMA
            elif price <= kama_aligned[i] * 0.99:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0