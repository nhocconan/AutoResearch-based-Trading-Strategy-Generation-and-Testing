#!/usr/bin/env python3
"""
Experiment #012: 12h Camarilla Pivot + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: Camarilla pivot levels (H3/L3) from 1d timeframe act as strong support/resistance 
on 12h chart. Long when price breaks above H3 with volume spike (>2.0x average) in trending 
market (choppiness index < 38.2). Short when price breaks below L3 with volume spike in 
trending market. Uses ATR-based stoploss (2.5x) and discrete position sizing (0.25). 
Designed for 12h timeframe to achieve 50-150 trades over 4 years with low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_012_12h_camarilla_pivot_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    # Typical price for pivot calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla pivot levels
    pivot = typical_price.rolling(window=1, min_periods=1).mean().values  # same as typical_price
    range_ = df_1d['high'] - df_1d['low']
    h3 = pivot + 1.1 * range_ / 2
    l3 = pivot - 1.1 * range_ / 2
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    
    # === HTF: 1w data for trend filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
    
    tr_chop = np.zeros(n)
    tr_chop[0] = high[0] - low[0]
    for i in range(1, n):
        tr_chop[i] = true_range(high[i], low[i], close[i-1])
    
    atr_sum_14 = pd.Series(tr_chop).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros(n)
    denominator = hh_14 - ll_14
    chop[14:] = 100 * np.log10(atr_sum_14[14:] / denominator[14:]) / np.log10(14)
    chop[:14] = 50.0  # neutral value for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for 1w EMA50 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Trending market only (CHOP < 38.2) ---
        trending_regime = chop[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = high[i] > h3_aligned[i-1]   # Break above H3
        breakout_down = low[i] < l3_aligned[i-1]  # Break below L3
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Trending regime + price > EMA50(1w) + breakout above H3 + volume spike
        if trending_regime and price > ema50_1w_aligned[i] and breakout_up and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Trending regime + price < EMA50(1w) + breakout below L3 + volume spike
        elif trending_regime and price < ema50_1w_aligned[i] and breakout_down and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals