#!/usr/bin/env python3
"""
Experiment #4936: 12h Camarilla Pivot + 1d Volume Spike + Choppiness Regime
HYPOTHESIS: On 12h timeframe, price reactions at 1d Camarilla pivot levels (L3, L4, H3, H4) with volume confirmation (>1.5x average) and choppiness regime filter (CHOP > 61.8 = ranging market) capture mean-reversion bounces in both bull and bear markets. Uses ATR(14) stoploss (2.0x) to manage risk. Designed for 12-37 trades/year on 12h timeframe to minimize fee drag while maintaining statistical significance. Works in ranging markets where pivot levels act as support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4936_12h_camarilla_pivot_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivots, volume MA, and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (using previous day's OHLC) ===
    if len(df_1d) >= 2:
        # Calculate pivots using previous day's data (shifted by 1 to avoid look-ahead)
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Camarilla levels
        l3 = pivot - (range_hl * 1.1 / 4.0)
        l4 = pivot - (range_hl * 1.1 / 2.0)
        h3 = pivot + (range_hl * 1.1 / 4.0)
        h4 = pivot + (range_hl * 1.1 / 2.0)
    else:
        pivot = np.full(len(df_1d), np.nan)
        l3 = np.full(len(df_1d), np.nan)
        l4 = np.full(len(df_1d), np.nan)
        h3 = np.full(len(df_1d), np.nan)
        h4 = np.full(len(df_1d), np.nan)
    
    # Align HTF Camarilla levels to 12h timeframe
    if len(pivot) > 0:
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
        l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
        h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    else:
        pivot_aligned = np.full(n, np.nan)
        l3_aligned = np.full(n, np.nan)
        l4_aligned = np.full(n, np.nan)
        h3_aligned = np.full(n, np.nan)
        h4_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Volume MA (20-period) for spike detection ===
    if len(df_1d) >= 20:
        vol_ma_1d = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_1d = np.full(len(df_1d), np.nan)
    
    if len(vol_ma_1d) > 0:
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Choppiness Index (CHOP) for regime filter ===
    if len(df_1d) >= 14:
        # True Range
        tr1 = df_1d['high'] - df_1d['low']
        tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
        tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
        ll_14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: CHOP = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
        # Avoid division by zero
        denominator = hh_14 - ll_14
        chop_raw = np.full(len(df_1d), np.nan)
        valid = (denominator > 0) & ~np.isnan(sum_tr_14) & ~np.isnan(denominator)
        if np.any(valid):
            chop_raw[valid] = 100 * np.log10(sum_tr_14[valid] / denominator[valid]) / np.log10(14)
    else:
        chop_raw = np.full(len(df_1d), np.nan)
    
    if len(chop_raw) > 0:
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    else:
        chop_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 14)  # Camarilla needs 2d, Vol MA 20, ATR 14
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x 1d volume MA
        vol_confirm = vol > 1.5 * vol_ma_1d_aligned[i]
        
        # Choppiness regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
        chop_regime = chop_aligned[i] > 61.8
        
        # --- Exit Logic (ATR trailing stop) ---
        if in_position:
            # Trailing stop: exit if price moves 2.0*ATR against position from entry
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long entry: price at or below L3/L4 with volume confirmation and chop regime
        long_entry = (price <= l3_aligned[i] or price <= l4_aligned[i]) and vol_confirm and chop_regime
        
        # Short entry: price at or above H3/H4 with volume confirmation and chop regime
        short_entry = (price >= h3_aligned[i] or price >= h4_aligned[i]) and vol_confirm and chop_regime
        
        # Final entry conditions
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals