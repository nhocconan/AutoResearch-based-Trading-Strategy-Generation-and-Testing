#!/usr/bin/env python3
"""
Experiment #4622: 12h Donchian(20) Breakout + Volume Spike + Chop Regime Filter
HYPOTHESIS: 12h price breaking 20-bar Donchian channels with volume >2x 20-bar MA in non-choppy markets (Choppiness Index < 38.2) captures strong trending moves. Uses 1d HTF for Donchian calculation and 1w HTF for chop regime to avoid look-ahead. Discrete sizing (0.25) and ATR trailing stop (2.5x) manage risk. Target: 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4622_12h_donchian20_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels from prior 1d data (shifted by 1 to avoid look-ahead)
    if len(df_1d) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        dh_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
        dl_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
        # Shift by 1 to use only completed periods
        dh_20 = np.concatenate([[np.nan], dh_20[:-1]])
        dl_20 = np.concatenate([[np.nan], dl_20[:-1]])
    else:
        dh_20 = np.array([])
        dl_20 = np.array([])
    
    # Precompute HTF: 1w data for Choppiness Index regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14-period) from prior 1w data
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w[1:] - low_1w[1:]
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index formula: 100 * log10(sum(tr)/(hh-ll)) / log10(14)
        # Avoid division by zero
        range_14 = hh_14 - ll_14
        chop_raw = np.zeros_like(tr_sum)
        mask = (range_14 > 0) & (~np.isnan(tr_sum))
        chop_raw[mask] = 100 * np.log10(tr_sum[mask] / range_14[mask]) / np.log10(14)
        
        # Shift by 1 to use only completed periods
        chop_raw = np.concatenate([[np.nan], chop_raw[:-1]])
    else:
        chop_raw = np.array([])
    
    # Align HTF indicators to 12h timeframe
    if len(dh_20) > 0:
        dh_20_aligned = align_htf_to_ltf(prices, df_1d, dh_20)
        dl_20_aligned = align_htf_to_ltf(prices, df_1d, dl_20)
    else:
        dh_20_aligned = np.full(n, np.nan)
        dl_20_aligned = np.full(n, np.nan)
        
    if len(chop_raw) > 0:
        chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw)
    else:
        chop_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Volume MA(20) for spike confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14)  # Donchian, Vol MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: spike >2x average for confirmation
        vol_spike = vol_ratio[i] > 2.0
        
        # Regime filter: non-choppy market (Choppiness Index < 38.2 = trending)
        trending_regime = chop_aligned[i] < 38.2
        
        # Breakout conditions: price breaks Donchian channels with volume spike in trending regime
        breakout_long = price > dh_20_aligned[i] and vol_spike and trending_regime
        breakout_short = price < dl_20_aligned[i] and vol_spike and trending_regime
        
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