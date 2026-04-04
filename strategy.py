#!/usr/bin/env python3
"""
Experiment #4665: 12h Donchian(20) Breakout + 1d ATR Filter + Volume Spike
HYPOTHESIS: 12h price breaking Donchian(20) channels (from prior 20 daily bars) 
with volume spike (>2x MA20) captures strong momentum. 1d ATR filter avoids 
entries during low volatility chop. Works in bull (breakouts) and bear 
(breakdowns) by trading both directions. Target: 12-37 trades/year on 12h TF.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4665_12h_donchian20_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Donchian(20) from prior 20 days ===
    if len(df_1d) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, df_1d['high'].values[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, df_1d['low'].values[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(len(df_1d), np.nan)
        donchian_low = np.full(len(df_1d), np.nan)
    
    # === 1d Indicators: ATR(14) for volatility filter ===
    if len(df_1d) >= 14:
        tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
        tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
        tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        # ATR ratio: current ATR / ATR MA(10) to detect expansion
        atr_ma = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
        atr_ratio = np.ones(len(atr_1d))
        atr_ratio[10:] = atr_1d[10:] / atr_ma[10:]
    else:
        atr_1d = np.full(len(df_1d), np.nan)
        atr_ratio = np.full(len(df_1d), np.nan)
    
    # Align HTF indicators to 12h timeframe
    if len(donchian_high) > 0:
        dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
        atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    else:
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
        atr_ratio_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20)  # Donchian and Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR(12h) below highest since entry (trailing stop)
                # Calculate 12h ATR on the fly for trailing (simplified: use price change)
                if price < highest_since_entry - 0.02 * highest_since_entry:  # 2% trailing stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2% above lowest since entry
                if price > lowest_since_entry + 0.02 * lowest_since_entry:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: spike for breakout confirmation (>2.0x MA20)
        vol_spike = vol_ratio[i] > 2.0
        
        # Volatility filter: only trade when ATR is expanding (>1.2x MA10)
        vol_expanding = atr_ratio_aligned[i] > 1.2
        
        # Breakout conditions: price breaks Donchian high/low with volume spike and volatility expansion
        breakout_long = price > dh_aligned[i] and vol_spike and vol_expanding
        breakout_short = price < dl_aligned[i] and vol_spike and vol_expanding
        
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