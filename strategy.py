#!/usr/bin/env python3
"""
Experiment #4677: 4h Donchian(20) Breakout + 1d Volume Spike + ATR Trailing Stop
HYPOTHESIS: 4h price breaking Donchian(20) channels (from prior 20 1d bars) with 1d volume confirmation (>2.0x average) captures momentum with controlled frequency.
Works in bull (breakouts) and bear (breakdowns) by being directionally agnostic. Target: 19-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4677_4h_donchian20_1d_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian and volume
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
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.ones(len(df_1d))
        vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_1d[20:]
    else:
        vol_ma_1d = np.full(len(df_1d), np.nan)
        vol_ratio_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF indicators to 4h timeframe
    if len(donchian_high) > 0:
        dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
        vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
        vol_ratio_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 14)  # Donchian, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation for breakouts (>2.0x 1d average volume)
        vol_breakout = vol_ratio_aligned[i] > 2.0
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation
        breakout_long = price > dh_aligned[i] and vol_breakout
        breakout_short = price < dl_aligned[i] and vol_breakout
        
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