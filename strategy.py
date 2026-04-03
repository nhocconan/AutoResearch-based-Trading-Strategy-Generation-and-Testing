#!/usr/bin/env python3
"""
Experiment #094: 1h Camarilla pivot + volume spike + chop regime filter

HYPOTHESIS: Camarilla pivot levels on 4h timeframe act as institutional support/resistance.
Long when price breaks above H3 with volume spike in low chop regime (trending).
Short when price breaks below L3 with volume spike in low chop regime.
Uses 1d for chop regime filter to avoid ranging markets. 1h only for entry timing.
Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
Works in bull (breakouts continue) and bear (breakdowns continue) via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    # === HTF: 4h data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels on 4h: based on previous day's OHLC
    # Need daily OHLC from 4h data - resample 4h to daily using actual boundaries
    # But we cannot resample - instead use 1d data directly for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from 1d OHLC (standard formula)
    if len(df_1d) >= 1:
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Camarilla levels: based on previous day's range
        # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
        # L3 = close - 1.25*(high-low), L4 = close - 1.5*(high-low)
        rng = high_1d - low_1d
        h3 = close_1d + 1.25 * rng
        l3 = close_1d - 1.25 * rng
        h4 = close_1d + 1.5 * rng
        l4 = close_1d - 1.5 * rng
        
        # Align to 1h timeframe (use previous day's levels)
        h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
        h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
        l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    else:
        h3_aligned = np.full(n, np.nan)
        l3_aligned = np.full(n, np.nan)
        h4_aligned = np.full(n, np.nan)
        l4_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for chop regime filter (Call ONCE before loop) ===
    # Choppy market indicator: high chop = ranging, low chop = trending
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        
        # Sum of TR over 14 periods
        atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Max high - min low over 14 periods
        hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Chop = LOG10(SUM(TR14)/(HH14-LL14)) * 100
        # Avoid division by zero
        range_14 = hh14 - ll14
        chop = np.zeros_like(close_1d)
        mask = (range_14 > 0) & (~np.isnan(range_14))
        chop[mask] = np.log10(atr14[mask] / range_14[mask]) * 100
        chop[~mask] = 50.0  # neutral when invalid
        
        # Align chop to 1h timeframe
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.ones(len(vol_1w))  # default to 1.0
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (chop < 40) ---
        # Chop < 38.2 = strongly trending, 38.2-61.8 = choppy, > 61.8 = ranging
        if chop_aligned[i] > 40.0:  # Avoid choppy/ranging markets
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1w ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.5
        
        # --- Entry Logic ---
        long_condition = (
            close[i] > h3_aligned[i] and 
            volume_spike
        )
        
        short_condition = (
            close[i] < l3_aligned[i] and 
            volume_spike
        )
        
        if long_condition:
            signals[i] = SIZE
        elif short_condition:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals