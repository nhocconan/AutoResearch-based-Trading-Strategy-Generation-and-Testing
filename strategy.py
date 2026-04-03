#!/usr/bin/env python3
"""
Experiment #023: 4h Camarilla Pivot + Volume Spike + Chop Regime Filter

HYPOTHESIS: Camarilla pivot levels from 1d timeframe act as strong support/resistance zones.
Price touching these levels with volume confirmation (>1.5x average) and in choppy markets 
(Choppiness Index > 61.8) indicates high-probability mean-reversion bounces. This strategy 
works in both bull and bear markets because it fades extreme moves at key levels rather than 
chasing trends. Targets 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to 
minimize fee drag while capturing reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_p = np.full(n, np.nan)
    
    if len(df_1d) >= 2:
        close_1d = df_1d['close'].values
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        for i in range(1, len(df_1d)):
            # Previous day's OHLC
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels
            camarilla_p[i] = (prev_high + prev_low + prev_close) / 3
            camarilla_h4[i] = camarilla_p[i] + range_ * 1.1 / 2
            camarilla_l4[i] = camarilla_p[i] - range_ * 1.1 / 2
            camarilla_h3[i] = camarilla_p[i] + range_ * 1.1 / 4
            camarilla_l3[i] = camarilla_p[i] - range_ * 1.1 / 4
            camarilla_h2[i] = camarilla_p[i] + range_ * 1.1 / 6
            camarilla_l2[i] = camarilla_p[i] - range_ * 1.1 / 6
            camarilla_h1[i] = camarilla_p[i] + range_ * 1.1 / 12
            camarilla_l1[i] = camarilla_p[i] - range_ * 1.1 / 12
        
        # Align to 4h timeframe (shifted by 1 for completed bars only)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
        camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
        camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
        camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
        camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    else:
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h2_aligned = np.full(n, np.nan)
        camarilla_l2_aligned = np.full(n, np.nan)
        camarilla_h1_aligned = np.full(n, np.nan)
        camarilla_l1_aligned = np.full(n, np.nan)
        camarilla_p_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Choppiness Index (14) for regime filter ===
    chop = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        
        # Avoid division by zero
        denominator = max_high - min_low
        chop_raw = 100 * np.log10(atr_sum / denominator) / np.log10(14)
        chop = np.where(denominator > 0, chop_raw, 50.0)  # Neutral when no range
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 200  # Ensure enough data for HTF and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in choppy markets (Choppiness > 61.8) ---
        choppy_market = chop[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price proximity to Camarilla levels (within 0.1%) ---
        def is_near_level(price, level):
            return abs(price - level) / level < 0.001
        
        near_h4 = is_near_level(close[i], camarilla_h4_aligned[i])
        near_l4 = is_near_level(close[i], camarilla_l4_aligned[i])
        near_h3 = is_near_level(close[i], camarilla_h3_aligned[i])
        near_l3 = is_near_level(close[i], camarilla_l3_aligned[i])
        near_h2 = is_near_level(close[i], camarilla_h2_aligned[i])
        near_l2 = is_near_level(close[i], camarilla_l2_aligned[i])
        near_h1 = is_near_level(close[i], camarilla_h1_aligned[i])
        near_l1 = is_near_level(close[i], camarilla_l1_aligned[i])
        
        # --- Exit Logic (Mean reversion: exit when price moves back toward pivot) ---
        if in_position:
            # Long exit: price reaches or crosses pivot from below
            if position_side > 0:
                if close[i] >= camarilla_p_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            # Short exit: price reaches or crosses pivot from above
            else:
                if close[i] <= camarilla_p_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price near L4/L3/L2/L1 + volume spike + choppy market
        long_condition = (near_l4 or near_l3 or near_l2 or near_l1) and volume_spike and choppy_market
        
        # Short: Price near H4/H3/H2/H1 + volume spike + choppy market
        short_condition = (near_h4 or near_h3 or near_h2 or near_h1) and volume_spike and choppy_market
        
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