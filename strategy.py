#!/usr/bin/env python3
"""
Experiment #028: 12h Camarilla Pivot + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: Camarilla pivot levels (L3, L4, H3, H4) derived from 1d OHLC act as institutional support/resistance. 
Price approaching these levels with volume spikes (>2x average) in a trending regime (Choppiness Index < 38.2) 
triggers mean-reversal bounces. The 12h timeframe provides sufficient noise reduction while capturing 
multi-day swings. Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. 
Works in both bull (buying dips at L3/L4) and bear (selling rallies at H3/H4) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_028_12h_camarilla_vol_chop_v1"
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
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(len(df_1d)):
        # Use previous completed 1d bar (shifted by 1 in align_htf_to_ltf)
        ph = df_1d['high'].iloc[i]
        pl = df_1d['low'].iloc[i]
        pc = df_1d['close'].iloc[i]
        rang = ph - pl
        if rang <= 0:
            continue
        camarilla_h3[i] = pc + (rang * 1.1 / 4)
        camarilla_l3[i] = pc - (rang * 1.1 / 4)
        camarilla_h4[i] = pc + (rang * 1.1 / 2)
        camarilla_l4[i] = pc - (rang * 1.1 / 2)
    
    # Align to 12h timeframe (with shift(1) for completed bars only)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === HTF: 1w data for Choppiness Index regime filter ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14) on 1w
    chop_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        tr_1w = np.zeros(len(df_1w))
        atr_1w = np.zeros(len(df_1w))
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        for i in range(1, len(df_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], 
                          abs(high_1w[i] - close_1w[i-1]), 
                          abs(low_1w[i] - close_1w[i-1]))
        
        # ATR(14) using Wilder's smoothing
        atr_1w[13] = np.mean(tr_1w[1:14])  # seed
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
        
        # Choppiness Index = 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
        for i in range(13, len(df_1w)):
            sum_atr = np.sum(atr_1w[i-13:i+1])
            max_high = np.max(high_1w[i-13:i+1])
            min_low = np.min(low_1w[i-13:i+1])
            if max_high > min_low:
                chop_1w[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
            else:
                chop_1w[i] = 50.0  # neutral
    
    # Align Choppiness Index to 12h
    chop_12h_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
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
    
    warmup = 200  # Ensure enough data for HTF indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(chop_12h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending market (Choppiness < 38.2) ---
        trending_regime = chop_12h_aligned[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Proximity to Camarilla Levels (within 0.5% of level) ---
        proximity_h3 = abs(close[i] - h3_12h[i]) / close[i] < 0.005
        proximity_l3 = abs(close[i] - l3_12h[i]) / close[i] < 0.005
        proximity_h4 = abs(close[i] - h4_12h[i]) / close[i] < 0.005
        proximity_l4 = abs(close[i] - l4_12h[i]) / close[i] < 0.005
        
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
        # Long: Near L3/L4 + volume spike + trending regime
        long_condition = (proximity_l3 or proximity_l4) and volume_spike and trending_regime
        
        # Short: Near H3/H4 + volume spike + trending regime
        short_condition = (proximity_h3 or proximity_h4) and volume_spike and trending_regime
        
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