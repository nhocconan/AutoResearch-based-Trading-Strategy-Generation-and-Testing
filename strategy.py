#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1_S1_Breakout_Volume_Regime_v1
12h timeframe strategy using daily Camarilla pivot levels (R1/S1) for breakout entries.
Entry conditions:
- Long: Price breaks above R1 with volume > 1.5x 20-period average
- Short: Price breaks below S1 with volume > 1.5x 20-period average
- Regime filter: Only trade when 1w ADX > 25 (trending market)
Exit: Price returns to pivot point (PP) or opposite Camarilla level (S1 for longs, R1 for shorts)
Position sizing: 0.25 (25% of capital)
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate pivot points from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R1 = Close + 1.1 * (High - Low)
    r1 = close_1d + 1.1 * (high_1d - low_1d)
    # S1 = Close - 1.1 * (High - Low)
    s1 = close_1d - 1.1 * (high_1d - low_1d)
    # R4 = Close + 1.1 * (High - Low) * 1.1/0.1 (simplified: R1 + 1.1*(High-Low))
    r4 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 0.1
    # S4 = Close - 1.1 * (High - Low) * 1.1/0.1
    s4 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 0.1
    
    # Align daily levels to 12h timeframe
    pp_aligned = align_ltf_to_htf(prices, df_1d, pp)
    r1_aligned = align_ltf_to_htf(prices, df_1d, r1)
    s1_aligned = align_ltf_to_htf(prices, df_1d, s1)
    r4_aligned = align_ltf_to_htf(prices, df_1d, r4)
    s4_aligned = align_ltf_to_htf(prices, df_1d, s4)
    
    # === Volume Filter: 20-period average volume ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly ADX for Regime Filter (trending market only) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # Calculate DM
    plus_dm_1w = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                          np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm_1w = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                           np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm_1w = np.concatenate([[0], plus_dm_1w])
    minus_dm_1w = np.concatenate([[0], minus_dm_1w])
    
    # ATR and ADX
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm_1w).rolling(window=14, min_periods=14).sum().values / (atr_1w * 14)
    minus_di_1w = 100 * pd.Series(minus_dm_1w).rolling(window=14, min_periods=14).sum().values / (atr_1w * 14)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to 12h
    adx_1w_aligned = align_ltf_to_htf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        if adx_1w_aligned[i] <= 25:
            # In ranging markets, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation
            if close[i] > r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below S1 with volume confirmation
            elif close[i] < s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price returns to PP or breaks below S1
            if close[i] <= pp_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to PP or breaks above R1
            if close[i] >= pp_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0