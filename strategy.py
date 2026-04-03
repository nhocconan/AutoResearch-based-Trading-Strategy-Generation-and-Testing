#!/usr/bin/env python3
"""
Experiment #162: 12h Camarilla Pivot + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: Camarilla pivot levels on daily timeframe act as strong support/resistance zones. 
Price touching these levels with volume confirmation (>2x average) and in favorable regimes 
(choppiness index > 61.8 for mean reversion or < 38.2 for trend continuation) captures 
high-probability reversals and continuations. Using 12h primary timeframe reduces trade 
frequency to target 50-150 total trades over 4 years, minimizing fee drag while allowing 
mean-reversion and trend strategies to work in both bull and bear markets.
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
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    camarilla_p = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        camarilla_p[i] = (h + l + c) / 3
        rng = h - l
        camarilla_h4[i] = camarilla_p[i] + rng * 1.1 / 2
        camarilla_l4[i] = camarilla_p[i] - rng * 1.1 / 2
        camarilla_h3[i] = camarilla_p[i] + rng * 1.1 / 4
        camarilla_l3[i] = camarilla_p[i] - rng * 1.1 / 4
        camarilla_h2[i] = camarilla_p[i] + rng * 1.1 / 6
        camarilla_l2[i] = camarilla_p[i] - rng * 1.1 / 6
        camarilla_h1[i] = camarilla_p[i] + rng * 1.1 / 12
        camarilla_l1[i] = camarilla_p[i] - rng * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 for completed bars only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # === HTF: 1w data for choppiness regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14) on weekly data
    chop_14 = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        atr_1w = np.zeros(len(df_1w))
        tr_1w = np.zeros(len(df_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(df_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], 
                          abs(high_1w[i] - close_1w[i-1]), 
                          abs(low_1w[i] - close_1w[i-1]))
        atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        for i in range(14, len(df_1w)):
            highest_high = np.max(high_1w[i-14:i])
            lowest_low = np.min(low_1w[i-14:i])
            sum_atr = np.sum(atr_1w[i-14:i])
            if sum_atr > 0:
                chop_14[i] = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
            else:
                chop_14[i] = 50.0  # Neutral when no range
    
    # Align Choppiness Index to 12h timeframe (shifted by 1 for completed bars only)
    chop_14_aligned = align_htf_to_ltf(prices, df_1w, chop_14)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    atr_14 = np.full(n, np.nan)
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
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF indicators and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop_14_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Choppiness Index ---
        chop_value = chop_14_aligned[i]
        is_ranging = chop_value > 61.8  # Mean reversion regime
        is_trending = chop_value < 38.2  # Trend continuation regime
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price Proximity to Camarilla Levels (within 0.2%) ---
        price = close[i]
        near_h4 = abs(price - camarilla_h4_aligned[i]) / price < 0.002
        near_l4 = abs(price - camarilla_l4_aligned[i]) / price < 0.002
        near_h3 = abs(price - camarilla_h3_aligned[i]) / price < 0.002
        near_l3 = abs(price - camarilla_l3_aligned[i]) / price < 0.002
        near_h2 = abs(price - camarilla_h2_aligned[i]) / price < 0.002
        near_l2 = abs(price - camarilla_l2_aligned[i]) / price < 0.002
        near_h1 = abs(price - camarilla_h1_aligned[i]) / price < 0.002
        near_l1 = abs(price - camarilla_l1_aligned[i]) / price < 0.002
        
        # --- Entry Logic ---
        # Long: Price near L3/L4 in ranging OR near L1/L2 in trending + volume spike
        long_condition = volume_spike and (
            (is_ranging and (near_l3 or near_l4)) or
            (is_trending and (near_l1 or near_l2))
        )
        
        # Short: Price near H3/H4 in ranging OR near H1/H2 in trending + volume spike
        short_condition = volume_spike and (
            (is_ranging and (near_h3 or near_h4)) or
            (is_trending and (near_h1 or near_h2))
        )
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
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