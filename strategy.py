#!/usr/bin/env python3
"""
Experiment #276: 12h Camarilla Pivot + 1d Volume Spike + Chop Regime

HYPOTHESIS: 12h Camarilla pivot levels (L3, L4, H3, H4) from 1d data act as strong support/resistance.
Entries occur on retests of these levels with volume confirmation (>1.5x average) and only when
the market is in a trending regime (Choppiness Index < 40). Exits use ATR-based stoploss (2.0 ATR)
and time-based exits (max 10 bars). The 12h timeframe targets 12-37 trades/year (50-150 total over 4 years)
to minimize fee drag. Works in bull markets (breakouts from H3/H4) and bear markets (breakdowns from L3/L4).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_276_12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and Choppiness Index (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d data
    def calculate_camarilla(high_arr, low_arr, close_arr):
        n1 = len(high_arr)
        pivot = np.full(n1, np.nan)
        resistance1 = np.full(n1, np.nan)
        resistance2 = np.full(n1, np.nan)
        resistance3 = np.full(n1, np.nan)
        resistance4 = np.full(n1, np.nan)
        support1 = np.full(n1, np.nan)
        support2 = np.full(n1, np.nan)
        support3 = np.full(n1, np.nan)
        support4 = np.full(n1, np.nan)
        
        for i in range(n1):
            if np.isnan(high_arr[i]) or np.isnan(low_arr[i]) or np.isnan(close_arr[i]):
                continue
            rng = high_arr[i] - low_arr[i]
            if rng <= 0:
                continue
            pivot[i] = (high_arr[i] + low_arr[i] + close_arr[i]) / 3.0
            resistance1[i] = close_arr[i] + rng * 1.1 / 12.0
            resistance2[i] = close_arr[i] + rng * 1.1 / 6.0
            resistance3[i] = close_arr[i] + rng * 1.1 / 4.0
            resistance4[i] = close_arr[i] + rng * 1.1 / 2.0
            support1[i] = close_arr[i] - rng * 1.1 / 12.0
            support2[i] = close_arr[i] - rng * 1.1 / 6.0
            support3[i] = close_arr[i] - rng * 1.1 / 4.0
            support4[i] = close_arr[i] - rng * 1.1 / 2.0
        
        return pivot, resistance1, resistance2, resistance3, resistance4, support1, support2, support3, support4
    
    # Calculate Choppiness Index for 1d data
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        n1 = len(high_arr)
        chop = np.full(n1, np.nan)
        if n1 < period:
            return chop
        atr = np.zeros(n1)
        for i in range(1, n1):
            atr[i] = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i-1]), abs(low_arr[i] - close_arr[i-1]))
        atr[0] = high_arr[0] - low_arr[0]
        atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        for i in range(period-1, n1):
            if atr_sum[i] == 0 or highest_high[i] <= lowest_low[i]:
                chop[i] = 50.0
            else:
                chop[i] = 100.0 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    # Calculate indicators on 1d data
    camarilla_pivots = calculate_camarilla(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Unpack Camarilla levels
    _, _, _, camarilla_h3, camarilla_h4, camarilla_l3, camarilla_l4, _, _ = camarilla_pivots
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF indicators, ATR, and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Chop Regime Filter: Only trade when market is trending (CHOP < 40) ---
        trending_regime = chop_1d_aligned[i] < 40.0
        
        # --- Proximity to Camarilla Levels (within 0.5% of level) ---
        proximity_h3 = abs(close[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] < 0.005
        proximity_h4 = abs(close[i] - camarilla_h4_aligned[i]) / camarilla_h4_aligned[i] < 0.005
        proximity_l3 = abs(close[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] < 0.005
        proximity_l4 = abs(close[i] - camarilla_l4_aligned[i]) / camarilla_l4_aligned[i] < 0.005
        
        # --- Exit Logic (ATR-based stoploss or time-based exit) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Time-based exit: max 10 bars in position
            if bars_since_entry >= 10:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Near H3/H4 level + volume spike + trending regime
        long_condition = (proximity_h3 or proximity_h4) and volume_spike and trending_regime
        
        # Short: Near L3/L4 level + volume spike + trending regime
        short_condition = (proximity_l3 or proximity_l4) and volume_spike and trending_regime
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals