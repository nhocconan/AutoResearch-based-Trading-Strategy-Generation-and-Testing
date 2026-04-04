#!/usr/bin/env python3
"""
Experiment #4657: 4h Donchian(20) Breakout + 1d Volume Spike + Choppiness Regime Filter
HYPOTHESIS: 4h price breaking Donchian(20) channels with volume confirmation (>2x average) 
captures strong momentum moves. Choppiness index (14) > 61.8 filters out ranging markets, 
allowing trades only in trending regimes. Works in bull (breakouts up) and bear (breakouts down).
Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4657_4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian and Choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Donchian(20) from prior 20 days ===
    if len(df_1d) >= 20:
        # Use prior 20 days' high/low (shifted by 1 to avoid look-ahead)
        ph = np.concatenate([[np.nan] * 20, df_1d['high'].values[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, df_1d['low'].values[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(len(df_1d), np.nan)
        donchian_low = np.full(len(df_1d), np.nan)
    
    # === 1d Indicators: Choppiness Index (14) ===
    if len(df_1d) >= 14:
        # True Range
        tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
        tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
        tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: 100 * log10(sum(tr)/ (hh - ll)) / log10(14)
        # Avoid division by zero
        range_hl = hh - ll
        chop = np.full(len(df_1d), np.nan)
        valid = (range_hl > 0) & (~np.isnan(tr_sum))
        chop[valid] = 100 * np.log10(tr_sum[valid] / range_hl[valid]) / np.log10(14)
    else:
        chop = np.full(len(df_1d), np.nan)
    
    # Align HTF indicators to 4h timeframe
    if len(donchian_high) > 0:
        dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
        chop_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(chop_aligned[i])):
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
        # Volume filter: confirmation for breakouts (>2x average)
        vol_breakout = vol_ratio[i] > 2.0
        
        # Regime filter: only trade in trending markets (Choppiness < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation and trending regime
        breakout_long = price > dh_aligned[i] and vol_breakout and trending_regime
        breakout_short = price < dl_aligned[i] and vol_breakout and trending_regime
        
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