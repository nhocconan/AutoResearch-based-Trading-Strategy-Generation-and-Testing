#!/usr/bin/env python3
"""
Experiment #082: 12h Camarilla pivot + volume spike + choppiness regime

HYPOTHESIS: On 12h timeframe, price touching Camarilla pivot levels (L3/H3) from prior 1d,
with 1d volume confirmation (>2.0x average) and choppiness regime filter (CHOP > 61.8 = range),
captures mean-reversion bounces in both bull and bear markets. The 12h timeframe reduces
overtrading vs lower timeframes while Camarilla levels provide precise entry/exit.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
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
    
    # Calculate Camarilla pivot levels for prior 1d
    if len(df_1d) >= 2:
        # Use prior completed 1d bar (yesterday)
        prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
        prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
        prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla levels
        h3 = pivot + (range_val * 1.1 / 4.0)
        l3 = pivot - (range_val * 1.1 / 4.0)
        h4 = pivot + (range_val * 1.1 / 2.0)
        l4 = pivot - (range_val * 1.1 / 2.0)
        
        # Align to LTF - each 12h bar gets prior day's levels
        h3_aligned = np.full(n, h3)
        l3_aligned = np.full(n, l3)
        h4_aligned = np.full(n, h4)
        l4_aligned = np.full(n, l4)
    else:
        h3_aligned = l3_aligned = h4_aligned = l4_aligned = np.full(n, np.nan)
    
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
    
    # === HTF: 1w data for choppiness regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (CHOP) on 1w
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w - low_1w
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1w[0] - low_1w[0]  # First period
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: CHOP = 100 * log10(sum(tr)/(hh-ll)) / log10(14)
        # Avoid division by zero
        denominator = hh - ll
        chop_raw = np.zeros_like(denominator)
        mask = denominator > 0
        chop_raw[mask] = 100 * np.log10(tr_sum[mask] / denominator[mask]) / np.log10(14)
        
        # For periods where hh == ll, set to 50 (neutral)
        chop_raw[~mask] = 50.0
        
        chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw)
    else:
        chop_aligned = np.full(n, 50.0)  # Neutral
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in ranging markets (CHOP > 61.8) ---
        is_ranging = chop_aligned[i] > 61.8
        
        if not is_ranging:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) on 1d ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic: Close position when price reaches opposite Camarilla level ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit when price reaches H3 (profit target) or breaks below L4 (stop)
                if close[i] >= h3_aligned[i] or close[i] <= l4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit when price reaches L3 (profit target) or breaks above H4 (stop)
                if close[i] <= l3_aligned[i] or close[i] >= h4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price touches L3 with volume confirmation in ranging market
        long_condition = (
            low[i] <= l3_aligned[i] and  # Price touches or goes below L3
            close[i] > l3_aligned[i] and  # But closes back above L3 (validation)
            volume_spike and
            is_ranging
        )
        
        # Short: Price touches H3 with volume confirmation in ranging market
        short_condition = (
            high[i] >= h3_aligned[i] and  # Price touches or goes above H3
            close[i] < h3_aligned[i] and  # But closes back below H3 (validation)
            volume_spike and
            is_ranging
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals