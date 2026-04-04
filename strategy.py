#!/usr/bin/env python3
"""
Experiment #4635: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking Donchian(20) channels from prior 1w with volume confirmation (>1.5x avg) and aligned with weekly pivot bias (above/below pivot) captures strong momentum breakouts. Weekly pivot provides higher-timeframe directional filter to avoid counter-trend trades. Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4635_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for weekly pivot and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    if len(df_1w) >= 1:
        # Use prior week's OHLC (shifted by 1 to avoid look-ahead)
        ph = np.concatenate([[np.nan], df_1w['high'].values[:-1]])  # prior week high
        pl = np.concatenate([[np.nan], df_1w['low'].values[:-1]])   # prior week low
        pc = np.concatenate([[np.nan], df_1w['close'].values[:-1]]) # prior week close
        
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (ph + pl + pc) / 3.0
        # Weekly R1 = 2*P - L, S1 = 2*P - H
        weekly_r1 = 2 * weekly_pivot - pl
        weekly_s1 = 2 * weekly_pivot - ph
    else:
        weekly_pivot = np.full(n, np.nan)
        weekly_r1 = np.full(n, np.nan)
        weekly_s1 = np.full(n, np.nan)
    
    # Calculate Donchian(20) from prior 1w OHLC (shifted by 1 to avoid look-ahead)
    if len(df_1w) >= 20:
        # Use prior 20 weeks' high/low (shifted by 1)
        ph_20 = np.concatenate([[np.nan] * 20, df_1w['high'].values[:-20]])  # prior 20 weeks high
        pl_20 = np.concatenate([[np.nan] * 20, df_1w['low'].values[:-20]])   # prior 20 weeks low
        
        # Rolling max/min of prior 20 weeks
        donchian_high = pd.Series(ph_20).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl_20).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Align HTF levels to 6h timeframe
    if len(df_1w) > 0:
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
        dh_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation for breakouts (>1.5x)
        vol_breakout = vol_ratio[i] > 1.5
        
        # Weekly pivot bias: long if price above pivot, short if below pivot
        pivot_bias_long = price > weekly_pivot_aligned[i]
        pivot_bias_short = price < weekly_pivot_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation AND pivot bias
        breakout_long = price > dh_aligned[i] and vol_breakout and pivot_bias_long
        breakout_short = price < dl_aligned[i] and vol_breakout and pivot_bias_short
        
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals