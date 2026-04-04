#!/usr/bin/env python3
"""
Experiment #4696: 12h Donchian(20) Breakout + 1d Volume Confirmation + ATR Stoploss
HYPOTHESIS: 12h price breaking Donchian(20) channels with volume confirmation (>1.3x avg volume) captures momentum while minimizing whipsaws. The Donchian(20) structure provides objective breakout levels that work in both bull (breakouts with volume) and bear (short breakdowns with volume) markets. Volume confirmation ensures breakouts are supported by participation. Target: 12-37 trades/year on 12h timeframe to avoid fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4696_12h_donchian20_1d_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF volume MA to 12h timeframe
    if len(vol_ma_1d) > 0:
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian(20) from prior 20 bars ===
    # Use prior 20 bars' high/low (shifted by 1 to avoid look-ahead)
    ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 bars high
    pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 bars low
    
    # Rolling max/min of prior 20 bars
    donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 14)  # Donchian, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation for breakouts (>1.3x)
        vol_breakout = volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i] and vol_breakout
        breakout_short = price < donchian_low[i] and vol_breakout
        
        # Final entry conditions: breakout + volume confirmation
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