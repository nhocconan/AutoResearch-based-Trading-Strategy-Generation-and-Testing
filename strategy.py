#!/usr/bin/env python3
"""
Experiment #065: 12h Camarilla pivot + volume spike + chop regime filter

HYPOTHESIS: 12h timeframe strategies have 54% keep rate in DB. Using Camarilla pivot levels from 1d HTF 
for structure, combined with 12h volume spike and choppiness regime filter, provides high-probability 
mean-reversion entries in ranging markets while avoiding false signals in strong trends. Targets 12-37 
trades/year (50-150 total over 4 years) with discrete position sizing to minimize fee drag. Works in 
both bull/bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        pivot = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        
        # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
        # H4 = close + range * 1.1/2
        # H3 = close + range * 1.1/4
        # H2 = close + range * 1.1/6
        # H1 = close + range * 1.1/12
        # L1 = close - range * 1.1/12
        # L2 = close - range * 1.1/6
        # L3 = close - range * 1.1/4
        # L4 = close - range * 1.1/2
        
        camarilla_h4 = close_1d + range_1d * 1.1 / 2.0
        camarilla_h3 = close_1d + range_1d * 1.1 / 4.0
        camarilla_h2 = close_1d + range_1d * 1.1 / 6.0
        camarilla_h1 = close_1d + range_1d * 1.1 / 12.0
        camarilla_l1 = close_1d - range_1d * 1.1 / 12.0
        camarilla_l2 = close_1d - range_1d * 1.1 / 6.0
        camarilla_l3 = close_1d - range_1d * 1.1 / 4.0
        camarilla_l4 = close_1d - range_1d * 1.1 / 2.0
        
        # For mean reversion: short near H3/H4, long near L3/L4
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 12h Indicators: Choppiness Index regime filter ===
    # Choppiness Index: higher = ranging market, lower = trending market
    chop = np.zeros(n)
    atr_period = 14
    chop_period = 14
    
    # Calculate True Range and ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate max/high-low range over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    range_sum = pd.Series(max_high - min_low).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Choppiness Index formula: 100 * log10(range_sum / (atr * chop_period)) / log10(chop_period)
    for i in range(chop_period, n):
        if atr[i] > 0 and range_sum[i] > 0:
            chop[i] = 100 * np.log10(range_sum[i] / (atr[i] * chop_period)) / np.log10(chop_period)
        else:
            chop[i] = 50.0  # Neutral
    
    # Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (avoid mean reversion)
    chop_ranging = chop > 61.8
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = max(100, chop_period)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in ranging markets (Chop > 61.8) ---
        if not chop_ranging[i]:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1d ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if not volume_spike:
            signals[i] = 0.0
            continue
        
        # --- Mean Reversion Entry Logic ---
        # Long: Price touches/below L3/L4 with volume spike in ranging market
        long_condition = (
            close[i] <= camarilla_l3_aligned[i] or 
            close[i] <= camarilla_l4_aligned[i]
        )
        
        # Short: Price touches/above H3/H4 with volume spike in ranging market
        short_condition = (
            close[i] >= camarilla_h3_aligned[i] or 
            close[i] >= camarilla_h4_aligned[i]
        )
        
        if long_condition:
            signals[i] = SIZE
        elif short_condition:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals